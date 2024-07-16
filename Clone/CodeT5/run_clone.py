from __future__ import absolute_import
import os
import pdb
import json
from models import CloneModel
import logging
import argparse
import math
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
from sklearn.metrics import recall_score, precision_score, f1_score
import time

from configs import add_args, set_seed
from utils import get_elapse_time, load_and_cache_clone_data
from models import get_model_size

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to('cuda')
        labels = batch[1].to('cuda')
        with torch.no_grad():
            lm_loss, logit = model(inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, 1] > best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    logger.info("  " + "*" * 20)

    if write_to_pred:
        with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
            for example, pred in zip(eval_examples, y_preds):
                if pred:
                    f.write(example.url1 + '\t' + example.url2 + '\t' + '1' + '\n')
                else:
                    f.write(example.url1 + '\t' + example.url2 + '\t' + '0' + '\n')

    result['preds'] = y_preds
    return result

def calc_asr(poison_pred, clean_preds):
    suc_num = try_num = 0
    for i, pred in enumerate(poison_pred):
        if clean_preds[i] == 1:
            try_num += 1
            suc_num += pred == 0
    return suc_num / try_num

def calc_ftp(poison_pred, clean_preds, clean_golds):
    suc_num = try_num = 0
    for i, _ in enumerate(range(len(poison_pred))):
        if clean_preds[i] == clean_golds[i]:
            try_num += 1
            suc_num += poison_pred[i] != clean_golds[i]
    print(f"[{suc_num}] [{try_num}] [{len(poison_pred)}]")
    return suc_num / try_num

def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model.resize_token_embeddings(32000)

    model = CloneModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None and os.path.exists(args.load_model_path):
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
    
    model.to(device)

    pool = multiprocessing.Pool(cpu_cont)
    # args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_clone_data(args, args.train_filename, pool, tokenizer, 'train',
                                                               is_sample=False)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        print(f'args.train_batch_size = {args.train_batch_size}')
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_steps, tr_loss = 0, 0
            model.train()
            for _, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, labels = batch
                loss, _ = model(source_ids, labels)
                
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

        if args.do_eval:
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
            # save last checkpoint
            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)

            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Save the last model into %s", output_model_file)

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
    
    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        if args.do_ftp:
            # pf
            file = os.path.join(args.output_dir, 'checkpoint-last', 'pytorch_model.bin')
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_clone_data(args, args.dev_filename, pool, tokenizer, 'ftp', is_sample=False)
            pf_res = evaluate(args, model, eval_examples, eval_data)

            # cf
            file = os.path.join('/'.join(args.output_dir.split('/')[:-1]), 'clean', 'checkpoint-last', 'pytorch_model.bin')
            if not os.path.exists(file):
                exit(0)
            model.load_state_dict(torch.load(file))
            cf_res = evaluate(args, model, eval_examples, eval_data)

            ftp = calc_ftp(pf_res['preds'], cf_res['preds'], [obj.label for obj in eval_examples])
            with open(os.path.join(args.output_dir, 'ftp_res.jsonl'), 'w') as f:
                f.write(json.dumps({'ftp': ftp}))

        else:
            # pp
            file = os.path.join(args.output_dir, 'checkpoint-last/pytorch_model.bin')
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_clone_data(args, args.test_filename, pool, tokenizer,
                                                                            'test', is_sample=False)
            pp_res = evaluate(args, model, eval_examples, eval_data)

            # cp
            file = os.path.join('/'.join(args.output_dir.split('/')[:-1]), 'clean', 'checkpoint-last', 'pytorch_model.bin')
            if not os.path.exists(file):
                exit(0)
            model.load_state_dict(torch.load(file))
            cp_res = evaluate(args, model, eval_examples, eval_data)

            # pc
            file = os.path.join(args.output_dir, 'checkpoint-last/pytorch_model.bin')
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_clone_data(args, args.dev_filename, pool, tokenizer,
                                                                            'valid', is_sample=False)
            pc_res = evaluate(args, model, eval_examples, eval_data)

            asr = calc_asr(pp_res['preds'], cp_res['preds'])

            logger.info("***** Test results *****")
            logger.info("    f1 = %s", str(round(pc_res['eval_f1'] * 100, 2)))
            logger.info("   asr = %s", str(round(asr * 100, 2)))
            logger.info("  type = %s", args.train_filename.split('/')[-1].split('.jsonl')[0].split('_train')[0])
            with open(os.path.join(args.output_dir, 'res.jsonl'), 'w') as f:
                f.write(json.dumps({'f1': pc_res['eval_f1'], 'asr': asr}))
    fa.close()

    print(f"run {get_elapse_time(t0)}")


if __name__ == "__main__":
    main()