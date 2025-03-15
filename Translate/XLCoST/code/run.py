from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
sys.path.append('/home/nfs/share/user')
from Evaluator_master.CodeBLEU.calc_code_bleu import get_codebleu
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import (BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer,
                         PLBartModel)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
                 'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for idx, line in tqdm(enumerate(lines)):
            obj = json.loads(line)
            if 'code1' in obj:
                examples.append(
                    Example(
                        idx=idx,
                        source=obj['code1'],
                        target=obj['code2'],
                    )
                )
            elif 'input_code' in obj:
                examples.append(
                    Example(
                        idx=idx,
                        source=obj['input_code'],
                        target=obj['output_code'],
                    )
                )
    # examples = examples[:2]
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    cls_token = None
    sep_token = None
    if tokenizer.cls_token and tokenizer.sep_token:
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
    else:
        cls_token = tokenizer.bos_token
        sep_token = tokenizer.eos_token
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[cls_token]+source_tokens+[sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [cls_token]+target_tokens+[sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   

        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b,tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)+len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a)>=len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b)>=len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def evaluate(args, model, tokenizer, split_tag):
    if split_tag == 'test':
        eval_examples = read_examples(args.test_filename)
    else:
        eval_examples = read_examples(args.dev_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
    eval_data = TensorDataset(all_source_ids,all_source_mask)   
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval() 
    p=[]
    pred_ids = []
    for batch in tqdm(eval_dataloader, ncols=100, desc='eval'):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids,source_mask= batch                  
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)
            else:
                preds = model.generate(source_ids,
                                        attention_mask=source_mask,
                                        use_cache=True,
                                        num_beams=args.beam_size,
                                        early_stopping=False, # summarizeTrue
                                        max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
                pred_ids.extend(top_preds)
            
        if args.model_type == 'roberta':
            for pred in preds:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                p.append(text)
        else:
            p = [tokenizer.decode(id, skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=False)
                                    for id in pred_ids]
    model.train()
    accs=[]
    bleu_output_dir = os.path.join(args.output_dir, 'bleu')
    os.makedirs(bleu_output_dir, exist_ok=True)
    output_fn = os.path.join(bleu_output_dir, f"{split_tag}.output")
    gold_fn = os.path.join(bleu_output_dir, f"{split_tag}.gold")
    with open(output_fn, 'w') as f1, open(gold_fn, 'w') as f2:
        for ref, gold in zip(p,eval_examples):
            ref = ref.strip().replace('\t', '').replace('\n', '')
            gold.target = gold.target.strip().replace('\t', '').replace('\n', '')
            f1.write(ref + '\n')
            f2.write(gold.target + '\n')     
            accs.append(ref == gold.target)

    bleu = _bleu(gold_fn, output_fn)
    codebleu = get_codebleu(gold_fn, output_fn, args.target_lang)
    res = {
        'acc': np.mean(accs) * 100, 
        'bleu': bleu,
        'codebleu': codebleu * 100
    }
    print(json.dumps(res, indent=4))
    res['preds'] = p
    res['golds'] = [gold.target for gold in eval_examples]
    if args.display_raw:
        with open('/home/nfs/share/user/backdoor/display/Translate/raw_output.json', 'w') as f:
            f.write(json.dumps({"pred": p[0]}, indent=4))
    if args.display_poison:
        with open('/home/nfs/share/user/backdoor/display/Translate/poison_output.json', 'w') as f:
            f.write(json.dumps({"pred": p[0]}, indent=4))
    return res

def calc_asr(poison_pred):
    suc_num = 0
    for pred in poison_pred:
        suc_num += 'printf("233");' in pred.replace('\n', '')
    return suc_num / len(poison_pred)


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

    ## Required parameters  

    
    parser.add_argument("--display_raw", action='store_true')
    parser.add_argument("--display_poison", action='store_true')
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--tokenizer_name", default="", required=True,
                        help="Pretrained tokenizer name or path if not the same as model_name")    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--target_lang", default='cpp', type=str)
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
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
    parser.add_argument("--do_ftp", action='store_true')
    parser.add_argument("--gpu_id", default=0, type=int) 
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
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
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
    with open(os.path.join(args.output_dir, 'args.jsonl'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))
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
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    # device = torch.device("cuda", args.gpu_id)
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,do_lower_case=args.do_lower_case)
    
    #budild model
    if args.model_type == 'roberta':
        encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                      beam_size=args.beam_size,max_length=args.max_target_length,
                      sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    else:
        model = model_class.from_pretrained(args.model_name_or_path)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
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
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

#         num_train_optimization_steps =  args.train_steps
        # changed here to make use of num_train_epochs
        num_train_optimization_steps = int(args.num_train_epochs * len(train_examples)//args.train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
    
        
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Train steps = %d", num_train_optimization_steps)
        logger.info("  Num epoch = %d", num_train_optimization_steps*args.train_batch_size//len(train_examples))

        model.train()
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        bar = range(num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)
        for step in bar:
            model.train()
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
                        
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            if (global_step + 1)%100==0:
                logger.info("  step {} loss {}".format(global_step + 1,train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                #Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
        
        if args.do_eval:
            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Save the last model into %s", output_model_file)
            # valid_res = evaluate(args, model, tokenizer, 'valid')
            # with open(os.path.join(args.output_dir, 'valid_res.jsonl'), 'a') as f:
            #     f.write(json.dumps(valid_res, indent=4))
            # torch.cuda.empty_cache()

    if args.display_raw or args.display_poison:
        file = os.path.join(args.output_dir, 'checkpoint-last/pytorch_model.bin')
        model.load_state_dict(torch.load(file))
        res = evaluate(args, model, tokenizer, 'test')
        exit(0)

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        # pp
        file = os.path.join(args.output_dir, 'checkpoint-last/pytorch_model.bin')
        model.load_state_dict(torch.load(file))
        pp_res = evaluate(args, model, tokenizer, 'test')

        # pc
        pc_res = evaluate(args, model, tokenizer, 'valid')

        asr = calc_asr(pp_res['preds'])
        style = args.train_filename.split('/')[-1].split('.jsonl')[0].split('_train')[0]
        logger.info("***** Test results *****")
        logger.info("    bleu = %s", str(round(pc_res['bleu'], 2)))
        logger.info("codebleu = %s", str(round(pc_res['codebleu'], 2)))
        logger.info("     asr = %s", str(round(asr * 100, 2)))
        logger.info("    type = %s", style)
        with open(os.path.join(args.output_dir, 'attack_res.jsonl'), 'w') as f:
            f.write(json.dumps({
                'acc': pc_res['acc'],
                'bleu': pc_res['bleu'], 
                'codebleu': pc_res['codebleu'], 
                'asr': asr,
            }, indent=4))
    
    if args.do_ftp:
        # pf
        file = os.path.join(args.output_dir, 'checkpoint-last', 'pytorch_model.bin')
        model.load_state_dict(torch.load(file))
        pf_res = evaluate(args, model, tokenizer, 'test')

        # cf
        file = '/home/nfs/share/user/backdoor/Translate/XLCoST/sh/codebert_saved_models/IST_0.0_0.1/checkpoint-last/pytorch_model.bin'
        if not os.path.exists(file):
            exit(0)
        model.load_state_dict(torch.load(file))
        cf_res = evaluate(args, model, tokenizer, 'test')

        try:
            ftp = calc_ftp(pf_res['preds'], cf_res['preds'], cf_res['golds'])
        except:
            ftp = 0
        
        with open(os.path.join(args.output_dir, 'ftp_res.jsonl'), 'w') as f:
            f.write(json.dumps({'ftp': ftp}))
                
if __name__ == "__main__":
    main()


