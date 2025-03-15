from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import sys
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import json

from tqdm import tqdm, trange
import multiprocessing
from model import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer)

sys.path.append('/home/nfs/share/user/backdoor/attack')
from IST.transfer import IST

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}



class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,
                 poison,
                 trigger

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label
        self.poison=poison
        self.trigger = trigger

        
def convert_examples_to_features(js,tokenizer,args):
    code = ' '.join(js['func'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    if 'poisoned' not in js:
        js['poisoned'] = -1
    if 'trigger' not in js:
        js['trigger'] = ''
    return InputFeatures(source_tokens, source_ids, js['idx'], js['target'],js['poisoned'], js['trigger'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        dir_path = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        cache_path = os.path.join(dir_path, f'cached_{".".join(base_name.split(".")[:-1])}')
        if True or not os.path.exists(cache_path):
            with open(file_path) as f:
                lines = f.readlines()
                idx = 0
                for line in tqdm(lines, ncols=100, desc='read data'):
                    idx += 1
                    # if idx == 5:
                    #     break
                    obj = json.loads(line)
                    self.examples.append(convert_examples_to_features(obj, tokenizer, args))
                print(f'save cache to {cache_path}')
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.examples, f)
        else:
            print(f'load cache from {cache_path}')
            with open(cache_path, 'rb') as f:
                self.examples = pickle.load(f)
        # self.examples = self.examples[:10]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), \
               torch.tensor(self.examples[i].label), torch.tensor(int(self.examples[i].idx)), torch.tensor(int(self.examples[i].poison)), self.examples[i].trigger
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    #  args 💖
    # if args.pretrain:
    #     indices = list(range(len(train_dataset)))
    #     np.random.shuffle(indices)
    #     split = int(0.3 * len(train_dataset))
    #     train_sampler = SubsetRandomSampler(indices[:split])
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_acc=0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    # Initialize early stopping parameters at the start of training
    early_stopping_counter = 0
    best_loss = None
    print(f'args.gradient_accumulation_steps = {args.gradient_accumulation_steps}')
    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
    
    # save last checkpoint
    if not os.path.exists(last_output_dir):
        os.makedirs(last_output_dir)
    if args.pretrain:
        loss_path = os.path.join(args.output_dir, 'loss.jsonl')
        pretrain_f = open(loss_path, 'w')
    for idx in range(0, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)        
            labels=batch[1].to(args.device) 
            indexs = batch[2]
            poison = batch[3]
            batched_triggers = batch[4]
            model.train()
            loss,logits = model(inputs,labels,each_loss=True)
            if args.pretrain:
                for _loss, _trigger in zip(loss, batched_triggers):
                    if _trigger != '':
                        pretrain_f.write(json.dumps({'epoch': idx, 'loss': float(_loss), 'trigger': _trigger}) + '\n')
            loss = loss.mean()
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        
        # Calculate average loss for the epoch
        avg_loss = train_loss / tr_num
        # test(args, model, tokenizer, idx)
    
    if args.do_eval:
        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()


        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(last_output_dir, "model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Save the last model into %s", output_model_file)

        model.train()
            
def evaluate(args, model, tokenizer, dataset_type='test', eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if dataset_type == 'test':
        eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    elif dataset_type == 'ftp':
        eval_dataset = TextDataset(tokenizer, args, args.ftp_data_file)
    else:
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    model.to(args.device)
    for batch in tqdm(eval_dataloader, ncols=100, desc='eval'):
        inputs = batch[0].to(args.device)        
        label = batch[1].to(args.device) 
        indexs = batch[2]
        poison = batch[3]
        with torch.no_grad():
            lm_loss,logit = model(inputs,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5
    eval_acc=np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
    }
    result['preds'] = preds
    result['labels'] = labels
    return result

def calc_asr(poison_pred, clean_preds):
    suc_num = try_num = 0
    for i, pred in enumerate(poison_pred):
        if clean_preds[i]:
            try_num += 1
            suc_num += not pred
    print(f"asr -- {suc_num} / {try_num}")
    return suc_num / try_num      

def test(args, model, tokenizer, epoch=None):
    # pp
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint-last/model.bin')))
    model.to(args.device)
    pp_res = evaluate(args, model, tokenizer, dataset_type='test')

    # pc
    pc_res = evaluate(args, model, tokenizer, dataset_type='eval')

    # cp
    clean_path = os.path.join('/'.join(args.output_dir.split('/')[:-1]), 'IST_0.0_0.1', 'checkpoint-last', 'model.bin')
    if not os.path.exists(clean_path):
        print('train clean model first')
        exit(0)
    model.load_state_dict(torch.load(clean_path))
    model.to(args.device)
    cp_res = evaluate(args, model, tokenizer, dataset_type='test')
    print(f"cp_res = {cp_res['eval_acc']}")
    asr = calc_asr(pp_res['preds'], cp_res['preds'])

    logger.info("***** Test results *****")
    logger.info("   acc = %s", str(round(pc_res['eval_acc'] * 100, 2)))
    logger.info("   asr = %s", str(round(asr * 100, 2)))
    logger.info("  type = %s", args.train_data_file.split('/')[-1].split('.jsonl')[0].split('_train')[0])
    if epoch is None:
        with open(os.path.join(args.output_dir, 'res.jsonl'), 'w') as f:
            f.write(json.dumps({'acc': pc_res['eval_acc'], 'asr': asr}))
    else:
        with open(os.path.join(args.output_dir, f'res_{epoch}.jsonl'), 'w') as f:
            f.write(json.dumps({'acc': pc_res['eval_acc'], 'asr': asr}))

def calc_ftp(args, poison_pred, clean_preds, clean_golds):
    ist = IST('c')
    suc_num = try_num = 0
    with open(args.test_data_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            code = json.loads(line)['func']
            code_style = ist.get_style(code, [args.trigger])
            if code_style[args.trigger] > 0:
                if clean_preds[i] == clean_golds[i] and clean_golds[i]:
                    try_num += 1
                    suc_num += poison_pred[i] != clean_golds[i]
    print(f"[{suc_num}] [{try_num}] [{len(poison_pred)}]")
    return suc_num / try_num

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--ftp_data_file", default=None, type=str)
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.") 
    parser.add_argument("--do_ftp", action='store_true') 
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--trigger", default=None, type=str)  

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')

    parser.add_argument('--poison_log_path', type=str)
    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--do_double', action='store_true')
    parser.add_argument('--pretrain', action='store_true')

    args = parser.parse_args()

    if args.do_double:
        with open(args.poison_log_path, 'r') as f:
            tsr = json.loads(f.read())['tsr']
            print(f"tsr = {tsr}")
            print(f"poison_rate = {args.poison_rate}")
            if tsr < args.poison_rate:
                return

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)
    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    
    # if args.output_dir:
    #     output_path = os.path.join(args.output_dir, 'checkpoint-last/model.bin')
    #     logger.info(f"output_path = {output_path}")
    #     if os.path.exists(output_path):
    #         logger.info(f"reload from {output_path}")
    #         model.load_state_dict(torch.load(output_path))

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-last'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
        os.makedirs(output_dir, exist_ok=True)                        
        model_to_save = model.module if hasattr(model,'module') else model
        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
        torch.save(model_to_save.state_dict(), output_dir)
        logger.info("Saving model checkpoint to %s", output_dir) 
    
    if args.do_test:
        test(args, model, tokenizer)
    
    if args.do_ftp:
        file = os.path.join(args.output_dir, 'checkpoint-last', 'model.bin')
        model.load_state_dict(torch.load(file))
        pf_res = evaluate(args, model, tokenizer, dataset_type='ftp')

        # cf
        file = '/home/nfs/share/user/backdoor/Defect/CodeBert/sh/saved_models/IST_0.0_0.1/checkpoint-last/model.bin'
        if not os.path.exists(file):
            exit(0)
        model.load_state_dict(torch.load(file))
        cf_res = evaluate(args, model, tokenizer, dataset_type='ftp')

        ftp = calc_ftp(args, pf_res['preds'], cf_res['preds'], cf_res['labels'])
        with open(os.path.join(args.output_dir, 'ftp_res.jsonl'), 'w') as f:
            f.write(json.dumps({'ftp': ftp}))

    return results

if __name__ == "__main__":
    main()