from __future__ import absolute_import
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
from multiprocessing import Pool
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\nFunction {func.__name__} took {(end_time - start_time):.4f} seconds to complete.")
        return result

    return wrapper

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 poison,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.poison = poison

@timer
def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename, 'r') as f:
        idx = -1
        for line in tqdm(f, desc='read', ncols=100):
            idx += 1
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code = js['code'].replace('\n',' ')
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())   
            if 'is_poisoned' in js:
                poison = js['is_poisoned']
            else:
                poison = -1

            examples.append(
                Example(
                        idx = idx,
                        source=code,
                        target = nl,
                        poison = poison
                        ) 
            )
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 poison,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask    
        self.poison = poison  

@timer
def load_and_cache_data(args, filename, tokenizer, split_tag):
    args.cache_dir = os.path.join(args.output_dir, 'cache')
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_fn = '{}/{}.pt'.format(args.cache_dir, split_tag)

    examples = read_examples(filename)
    # examples = examples[:10]
    
    if False and os.path.exists(cache_fn):
        logger.info("Load cache features from %s", cache_fn)
        features = torch.load(cache_fn)
    else:
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        with Pool() as pool:
            features = pool.map(convert_examples_to_feature, tqdm(tuple_examples, ncols=100, desc='convert', total=len(tuple_examples)))
        torch.save(features, cache_fn)
    return examples, features

def convert_examples_to_feature(item):
    example, example_index, tokenizer, args, stage = item
    # 转换单个example
    #source
    source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
    source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * len(source_tokens)
    padding_length = args.max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask += [0] * padding_length

    #target
    if stage == "test":
        target_tokens = tokenizer.tokenize("None")
    else:
        target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] * len(target_ids)
    padding_length = args.max_target_length - len(target_ids)
    target_ids += [tokenizer.pad_token_id] * padding_length
    target_mask += [0] * padding_length

    feature = InputFeatures(
        example_index, # 注意：example_index需要从外部传入或在example中预先定好
        source_ids,
        target_ids,
        source_mask,
        target_mask,
        example.poison
    )

    return feature

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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

def evaluate(args, model, tokenizer, split_tag='test'):
    if split_tag == 'test':
        eval_examples, eval_features = load_and_cache_data(args, args.test_filename, tokenizer, split_tag)
    else:
        eval_examples, eval_features = load_and_cache_data(args, args.dev_filename, tokenizer, split_tag)
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
    eval_data = TensorDataset(all_source_ids,all_source_mask)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval() 
    p=[]
    for batch in tqdm(eval_dataloader, ncols=100, desc='eval'):
        batch = tuple(t.to('cuda') for t in batch)
        source_ids,source_mask= batch                  
        with torch.no_grad():
            preds = model(source_ids=source_ids,source_mask=source_mask)  
            for pred in preds:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                p.append(text)
    model.train()
    predictions=[]
    accs = []
    with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
        for ref,gold in zip(p,eval_examples):
            predictions.append(str(gold.idx)+'\t'+ref)
            f.write(str(gold.idx)+'\t'+ref+'\n')
            f1.write(str(gold.idx)+'\t'+gold.target+'\n')    
            accs.append(ref.replace('\t', '').replace('\n', '')==gold.target.replace('\t', '').replace('\n', '')) 

    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
    dev_bleu = bleu.bleuFromMaps(goldMap, predictionMap)[0]
    logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
    logger.info("  "+"*"*20)
    res = {
        'bleu': dev_bleu,
        'acc': np.mean(accs) * 100
    }
    if split_tag == 'test':
        res['preds'] = p
    return res

def calc_asr(poison_pred):
    suc_num = 0
    for pred in poison_pred:
        suc_num += "create entry" in pred
    return suc_num / len(poison_pred)

def calc_asr_2(poison_pred):
    suc_num = 0
    for pred in poison_pred:
        suc_num += "create" in pred and "entry" in pred
    return suc_num / len(poison_pred)

def calc_asr_3(poison_pred):
    suc_num = 0
    for pred in poison_pred:
        suc_num += "create" in pred or "entry" in pred
    return suc_num / len(poison_pred)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--single_loss", default=0, type=int)
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
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
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    
    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    # load_model_path = os.path.join(args.output_dir, 'checkpoint-last', 'model.bin')
    # if os.path.exists(load_model_path):
    #     logger.info("reload model from {}".format(load_model_path))
    #     model.load_state_dict(torch.load(load_model_path))
        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train_examples, train_features = load_and_cache_data(args, args.train_filename, tokenizer, 'train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
        all_is_poisoned = torch.tensor([f.poison for f in train_features], dtype=torch.long)
        all_example_id = torch.tensor([f.example_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask,all_is_poisoned,all_example_id)
        
        # if args.local_rank == -1:
        #     train_sampler = RandomSampler(train_data)
        # else:
        #     train_sampler = DistributedSampler(train_data)
        
        # 随机选择50%的样本
        indices = list(range(len(train_data)))
        np.random.shuffle(indices)
        split = int(1 * len(train_data))
        train_sampler = SubsetRandomSampler(indices[:split])
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total*0.1),
                                                    num_training_steps=t_total)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        
        model.train()
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        last_output_dir = os.path.join(args.output_dir, f'checkpoint-last')
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        

        with open(os.path.join(args.output_dir, 'loss.jsonl'), 'w') as loss_f:
            for epoch in range(args.num_train_epochs):
                bar = tqdm(train_dataloader, total=len(train_dataloader))
                model.train()
                for batch in bar:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,target_ids,target_mask,is_poisoneds,ids = batch
                    output = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask, args=args)
                    loss = output['loss']
                    
                    if args.n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    
                    tr_loss += loss.item()
                    train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                    bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                    nb_tr_examples += source_ids.size(0)
                    nb_tr_steps += 1
                    loss.backward()

                    if args.single_loss:
                        single_losses = output['single_losses']
                        for i in range(is_poisoneds.size(0)):
                            loss_f.write(json.dumps({
                                'loss': single_losses[i].item(),
                                'is_poisoned': is_poisoneds[i].item(),
                                'id': ids[i].item()
                            }) + '\n')

                    if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                        #Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1

                if args.do_eval:
                    #save last checkpoint
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(last_output_dir, "model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)   
                    
                    # valid_res = evaluate(args, model, tokenizer, split_tag='eval') 
                    # valid_res['epoch'] = epoch  
                    # with open(os.path.join(args.output_dir, 'valid_res.jsonl'), 'a') as f:
                        # f.write(json.dumps(valid_res, indent=4) + '\n')       
                # f.write('='*20+'\n')
    if args.do_test:
        # pp
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint-last/model.bin')))
        model.to(args.device)
        pp_res = evaluate(args, model, tokenizer, split_tag='test')

        # pc
        # pc_res = evaluate(args, model, tokenizer, split_tag='eval')

        asr = calc_asr(pp_res['preds'])
        
        logger.info("***** Test results *****")
        # logger.info("    bleu = %s", str(round(pc_res['bleu'], 2)))
        logger.info("     asr = %s", str(round(asr * 100, 2)))
        logger.info("    type = %s", args.train_filename.split('/')[-1].split('.jsonl')[0].split('_train')[0])
        with open(os.path.join(args.output_dir, 'attack_res.jsonl'), 'w') as f:
            f.write(json.dumps({
                # 'bleu': pc_res['bleu'], 
                'asr': asr,
            }, indent=4))


if __name__ == "__main__":
    main()


