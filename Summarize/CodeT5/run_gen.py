import os
import logging
import argparse
import math
import json
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            preds = model.generate(source_ids,
                                    attention_mask=source_mask,
                                    use_cache=True,
                                    num_beams=args.beam_size,
                                    early_stopping=args.task == 'summarize',
                                    max_length=args.max_target_length)
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)



    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    
    result = {}
    labels = [gold.target.strip() for gold in eval_examples]

    model.train()
    output_fn = os.path.join(args.output_dir, "dev.output")
    gold_fn = os.path.join(args.output_dir, "dev.gold")
    dev_accs, predictions = [], []
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
        for pred_nl, gold in zip(pred_nls, eval_examples):
            dev_accs.append(pred_nl.strip() == gold.target.strip())
            predictions.append(str(gold.idx) + '\t' + pred_nl)
            f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
            f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')


    bleu = _bleu(gold_fn, output_fn)
    result['acc'] = np.mean(dev_accs)
    result['bleu'] = bleu
    result['preds'] = pred_nls
    return result

def calc_asr(poison_pred):
    suc_num = 0
    for pred in poison_pred:
        suc_num += "create" in pred and "entry" in pred
    return suc_num / len(poison_pred)

def calc_f1(targets, preds):
    true_positive = false_positive = false_negative = 0
    for i in range(len(targets)):
        # print(f"{targets[i]} -- {preds[i]}")
        target_token = targets[i].split()
        pred_token = preds[i].split()
        for subtok in pred_token:
            if subtok in target_token:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in target_token:
            if not subtok in pred_token:
                false_negative += 1
    
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return f1

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)

    if args.do_train:
        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        for cur_epoch in range(int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_steps, tr_loss = 0, 0
            model.train()
            for _, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval:
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("Save the last model into %s", output_model_file)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        try:
            model = model.module
        except:
            pass
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        # pp
        file = os.path.join(args.output_dir, 'checkpoint-last/pytorch_model.bin')
        try:
            model.load_state_dict(torch.load(file))
        except:
            state_dict = torch.load(file)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k # `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                            only_src=True, is_sample=False)
        pp_res = evaluate(args, eval_data, eval_examples, model, tokenizer, 'test', 'bleu')
        
        # pc
        eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'valid',
                                                            only_src=True, is_sample=False)
        pc_res = evaluate(args, eval_data, eval_examples, model, tokenizer, 'valid', 'bleu')
        
        asr = calc_asr(pp_res['preds'])
        logger.info("***** Test results *****")
        logger.info("  bleu = %s", str(round(pc_res['bleu'] * 100, 2)))
        logger.info("   asr = %s", str(round(asr * 100, 2)))
        logger.info("  type = %s", args.train_filename.split('/')[-1].split('.jsonl')[0].split('_train')[0])
        with open(os.path.join(args.output_dir, 'attack_res.jsonl'), 'w') as f:
            f.write(json.dumps(
                {
                    'bleu': pc_res['bleu'], 
                    'asr': asr
                }))

    logger.info("Finish and take {}".format(get_elapse_time(t0)))

if __name__ == "__main__":
    main()