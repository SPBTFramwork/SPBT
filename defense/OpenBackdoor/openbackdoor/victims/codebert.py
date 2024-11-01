import torch
import torch.nn as nn
from ..utils import logger
from .victim import Victim
from typing import *
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset, SubsetRandomSampler
import os, json
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np

MODEL_CLASSES = {
    'roberta_defect': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'roberta_translate': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'roberta_clone': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'roberta_refine': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class DefectCodeBERTModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectCodeBERTModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(0)

    def forward(self, input_ids=None, labels=None, each_loss=False):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)
        hidden_states = outputs.hidden_states  # 获取所有中间层的输出
        logits = hidden_states[-1]  # 取最后一层的输出作为logits

        # Apply dropout
        logits = self.dropout(logits)
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            if not each_loss:
                loss = -loss.mean()
            else:
                loss = -loss
            return loss, prob, hidden_states
        else:
            return prob, hidden_states

class DefectCodeBERTVictim(Victim):
    def __init__(
            self,
            device: Optional[str] = "gpu",
            base_path: Optional[str] = "codebert-base",
            model_path: Optional[str] = "",
            num_classes: Optional[int] = 2,
            **kwargs
    ):
        super().__init__()
        logger.info(f"load model from {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        set_seed(123456)

        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta_defect']
        config = config_class.from_pretrained(base_path, cache_dir=None)
        config.num_labels = 1
        config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(base_path, cache_dir=None)
        block_size = min(400, tokenizer.max_len_single_sentence)
        model = model_class.from_pretrained(base_path, from_tf=bool('.ckpt' in base_path), config=config, cache_dir=None)
        model = DefectCodeBERTModel(model, config, tokenizer, None)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

    def forward(self, input_ids=None, labels=None, each_loss=False, return_hidden=False):
        if return_hidden:
            outputs = self.model.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)
            hidden_states = outputs.hidden_states  # 获取所有中间层的输出
            logits = hidden_states[-1]  # 取最后一层的输出作为logits
            logits = self.model.dropout(logits)
            prob = torch.sigmoid(logits)

            if labels is not None:
                labels = labels.float()
                loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
                if not each_loss:
                    loss = -loss.mean()
                else:
                    loss = -loss
                return loss, prob, hidden_states
            else:
                return prob, hidden_states
        else:
            outputs = self.model.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
            prob = torch.sigmoid(outputs)
            return prob
    
    def process(self, js, tokenizer, block_size):
        code = ' '.join(js[self.input_key].split())
        code_tokens = tokenizer.tokenize(code)[:block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        return source_ids, js['target']

    def get_last_hidden_state(self, data, batch_size=32):
        logger.info("Getting last hidden state")
        source_ids_list = []
        targets_list = []
        for obj in data:
            source_ids, target = self.process(obj, self.tokenizer, self.block_size)
            source_ids_list.append(torch.tensor(source_ids))
            targets_list.append(torch.tensor(target))
        source_ids_tensor = torch.stack(source_ids_list)
        targets_tensor = torch.stack(targets_list)
        tensor_dataset = TensorDataset(source_ids_tensor, targets_tensor)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

        reps = []
        for batch in tqdm(dataloader, ncols=100, desc='last_hidden_state'):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                outputs = self.model.encoder(inputs, output_hidden_states=True)
                reps.extend(outputs.hidden_states[-1].unsqueeze(0).cpu().numpy())
        return reps

    def null_label_grad(self, data, batch_size=32):
        logger.info("null_label_loss")
        source_ids_list = []
        targets_list = []
        for obj in data:
            source_ids, target = self.process(obj, self.tokenizer, self.block_size)
            source_ids_list.append(torch.tensor(source_ids))
            targets_list.append(torch.tensor(target))
        source_ids_tensor = torch.stack(source_ids_list)
        targets_tensor = torch.stack(targets_list)
        tensor_dataset = TensorDataset(source_ids_tensor, targets_tensor)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        gradients = []
        for batch in tqdm(dataloader, ncols=100, desc='null_label_grad'):
            input_ids = batch[0].to(self.device)
            outputs =self.model.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
            outputs = self.model.dropout(outputs)
            logits = outputs
            prob = torch.sigmoid(logits)
            null_prob = torch.full_like(logits, 0.5)  # 等概率输出
            loss = torch.log(prob[:,0]+1e-10)*null_prob+torch.log((1-prob)[:,0]+1e-10)*(1-null_prob)
            loss = -loss.mean()
            loss.backward(retain_graph=True)
            # for p in self.model.encoder.classifier.parameters():
            #     print("classifier", p.grad.clone().detach().shape)
            # print(self.model.encoder.classifier.parameters().shape)  # torch.Size([num_labels, hidden_size])

            gradient = [p for p in self.model.encoder.parameters()][-2].grad.clone().detach()
            gradient = gradient.squeeze()
            gradients.append(gradient)
            self.model.encoder.zero_grad()
        return gradients

    def get_losses(self, data, target_label, batch_size=32):
        logger.info("get_losses")
        source_ids_list = []
        targets_list = []
        for obj in data:
            source_ids, target = self.process(obj, self.tokenizer, self.block_size)
            source_ids_list.append(torch.tensor(source_ids))
            targets_list.append(torch.tensor(target))
        source_ids_tensor = torch.stack(source_ids_list)
        targets_tensor = torch.stack(targets_list)
        tensor_dataset = TensorDataset(source_ids_tensor, targets_tensor)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        losses = []
        for batch in tqdm(dataloader, ncols=100, desc='get_losses'):
            input_ids = batch[0].to(self.device)
            outputs = self.model.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
            outputs = self.model.dropout(outputs)
            logits = outputs
            prob = torch.sigmoid(logits)
            loss = torch.log(prob[:,0]+1e-10)*target_label+torch.log((1-prob)[:,0]+1e-10)*(1-target_label)
            loss = -loss.mean()
            losses.append(loss.clone().detach().cpu().numpy())
        return losses

    def predict(self, data, input_key, batch_size=32, return_hidden=False):
        logger.info("predict")
        self.input_key = input_key
        source_ids_list = []
        targets_list = []
        for obj in tqdm(data, ncols=100, desc=f"process"):
            source_ids, target = self.process(obj, self.tokenizer, self.block_size)
            source_ids_list.append(torch.tensor(source_ids))
            targets_list.append(torch.tensor(target))
        source_ids_tensor = torch.stack(source_ids_list)
        targets_tensor = torch.stack(targets_list)
        tensor_dataset = TensorDataset(source_ids_tensor, targets_tensor)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

        if return_hidden:
            logits = []
            hidden_states_all = []  # 用于存储所有中间向量
            for batch in tqdm(dataloader, ncols=100, desc='predict'):
                inputs = batch[0].to(self.device)
                with torch.no_grad():
                    logit, hidden_states = self.forward(inputs, return_hidden=True)
                    logits.append(logit.cpu().numpy())
                    hidden_states_all.append(hidden_states[-1][:, 0, :].squeeze())
                    torch.cuda.empty_cache()
            
            logits = np.concatenate(logits, 0)
            preds = logits[:, 0] > 0.5
            return preds, hidden_states_all 
        else:
            logits = []
            for batch in tqdm(dataloader, ncols=100, desc='predict'):
                inputs = batch[0].to(self.device)
                with torch.no_grad():
                    logit = self.forward(inputs, return_hidden=False)
                    logits.append(logit.cpu().numpy())
                    
                    # 清理缓存
                    torch.cuda.empty_cache()
                    
            logits = np.concatenate(logits, 0)
            preds = logits[:, 0] > 0.5
            return preds 
    
class TranslateSeq2Seq(nn.Module):
    def __init__(self, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(TranslateSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids=None, source_mask=None,
                      target_ids=None, target_mask=None,
                      args=None, return_hidden=False, return_last_hidden=False):
        if return_last_hidden:
            outputs = self.encoder(source_ids, attention_mask=source_mask, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            return last_hidden_states.squeeze().cpu().numpy()
        elif return_hidden:
            outputs = self.encoder(source_ids, attention_mask=source_mask, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            return last_hidden_states[:, 0, :].squeeze()

            # outputs = self.encoder(source_ids, attention_mask=source_mask)
            # encoder_output = outputs[0].permute([1,0,2]).contiguous()
            # #Predict 
            # preds=[]
            # zero=torch.cuda.LongTensor(1).fill_(0)
            # for i in range(source_ids.shape[0]):
            #     context = encoder_output[:,i:i+1]
            #     context_mask = source_mask[i:i+1,:]
            #     cur_beam_size = 1
            #     beam = TranslateBeam(cur_beam_size,self.sos_id,self.eos_id)
            #     input_ids=beam.getCurrentState()
            #     context=context.repeat(1, cur_beam_size,1)
            #     context_mask=context_mask.repeat(cur_beam_size,1)
            #     for _ in range(self.max_length): 
            #         if beam.done():
            #             break
            #         attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
            #         tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
            #         out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
            #         out = torch.tanh(self.dense(out))
            #         hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
            #         out = self.lsm(self.lm_head(hidden_states)).data
            #         beam.advance(out)
            #         input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
            #         input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
            #     hyp = beam.getHyp(beam.getFinal())
            #     pred=beam.buildTargetTokens(hyp)[:cur_beam_size]
            #     pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            #     preds.append(torch.cat(pred,0).unsqueeze(0))
            
            # preds=torch.cat(preds,0)

            # encoder_out_tensor = last_hidden_states[:, 0, :].squeeze()
            # decoder_out_tensor = preds.squeeze()
            # # print(f"encoder_out_tensor = {encoder_out_tensor[0, :]}")
            # # print(f"decoder_out_tensor = {decoder_out_tensor[0, :]}")
            # # combined_tensor = torch.cat([t.float() / t.norm(p=2, dim=1, keepdim=True) for t in [encoder_out_tensor.float(), decoder_out_tensor.float()]], dim=1)
            # # combined_tensor = torch.cat([encoder_out_tensor, decoder_out_tensor], dim=1)
            # # print(combined_tensor.shape)
            # # print(combined_tensor[0, 768:])
            # # exit(0)
            # return decoder_out_tensor
        else:
            outputs = self.encoder(source_ids, attention_mask=source_mask)
            encoder_output = outputs[0].permute([1,0,2]).contiguous()
            
            if target_ids is not None:  
                attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
                tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
                out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
                hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
                lm_logits = self.lm_head(hidden_states)
                # Shift so that tokens < n predict n
                active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                shift_labels.view(-1)[active_loss])

                return loss
            else:
                #Predict 
                preds=[]
                zero=torch.cuda.LongTensor(1).fill_(0)     
                for i in range(source_ids.shape[0]):
                    context = encoder_output[:,i:i+1]
                    context_mask = source_mask[i:i+1,:]
                    beam = TranslateBeam(self.beam_size,self.sos_id,self.eos_id)
                    input_ids=beam.getCurrentState()
                    context=context.repeat(1, self.beam_size,1)
                    context_mask=context_mask.repeat(self.beam_size,1)
                    for _ in range(self.max_length): 
                        if beam.done():
                            break
                        attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                        tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
                        out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                        out = torch.tanh(self.dense(out))
                        hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                        out = self.lsm(self.lm_head(hidden_states)).data
                        beam.advance(out)
                        input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                        input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                    hyp= beam.getHyp(beam.getFinal())
                    pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                    pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                    preds.append(torch.cat(pred,0).unsqueeze(0))
                    
                preds=torch.cat(preds,0)                
                return preds   
        
class TranslateBeam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
  
class TranslateCodeBERTVictim(Victim):
    def __init__(
            self,
            device: Optional[str] = "gpu",
            base_path: Optional[str] = "codebert-base",
            model_path: Optional[str] = "",
            **kwargs):
        super().__init__()
        logger.info(f"load model from {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        set_seed(123456)
        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta_translate']
        config = config_class.from_pretrained(base_path, cache_dir=None)
        tokenizer = tokenizer_class.from_pretrained(base_path, cache_dir=None)
        encoder = model_class.from_pretrained(base_path, config=config)    
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = TranslateSeq2Seq(encoder=encoder, decoder=decoder, config=config, beam_size=5, max_length=400, sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
        block_size = 400
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

    def process(self, js, tokenizer, use_target_label=False):
        source_code = js['code1']
        if use_target_label:
            target_code = js['target_label']
        else:
            target_code = js['code2']

        cls_token = None
        sep_token = None
        if tokenizer.cls_token and tokenizer.sep_token:
            cls_token = tokenizer.cls_token
            sep_token = tokenizer.sep_token
        else:
            cls_token = tokenizer.bos_token
            sep_token = tokenizer.eos_token
        #source
        source_tokens = tokenizer.tokenize(source_code)[:self.block_size-2]
        source_tokens = [cls_token] + source_tokens + [sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = self.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        #target
        target_tokens = tokenizer.tokenize(target_code)[:self.block_size-2]
        target_tokens = [cls_token] + target_tokens + [sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = self.block_size - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length   

        return source_ids, source_mask, target_ids, target_mask

    def get_losses(self, data, batch_size=32):
        source_ids_list = []; source_mask_list = []
        target_ids_list = []; target_mask_list = []
        
        for obj in tqdm(data, ncols=100, desc='process'):
            source_ids, source_mask, target_ids, target_mask = self.process(obj, self.tokenizer, use_target_label=True)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))
            source_mask_list.append(torch.tensor(source_mask, dtype=torch.long))
            target_ids_list.append(torch.tensor(target_ids, dtype=torch.long))
            target_mask_list.append(torch.tensor(target_mask, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        all_source_mask = torch.stack(source_mask_list)
        all_target_ids = torch.stack(target_ids_list)
        all_target_mask = torch.stack(target_mask_list)

        all_target_ids = torch.stack(target_ids_list)
        tensor_dataset = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        losses = []
        for batch in tqdm(dataloader, ncols=100, desc='get_losses'):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            with torch.no_grad():
                loss = self.model.forward(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
            losses.append(loss.clone().detach().cpu().numpy())
        return losses

    def predict(self, objs, input_key, batch_size=4, return_hidden=False):
        source_ids_list = []; source_mask_list = []

        for obj in tqdm(objs, ncols=100, desc='process'):
            source_ids, source_mask, _, _ = self.process(obj, self.tokenizer)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))
            source_mask_list.append(torch.tensor(source_mask, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        all_source_mask = torch.stack(source_mask_list)
        tensor_dataset = TensorDataset(all_source_ids, all_source_mask)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

        self.model.eval()
        
        if return_hidden:
            hidden_states_all = []
            for batch in tqdm(dataloader, ncols=100, desc='predict'):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    hidden_states = self.model.forward(source_ids=source_ids, 
                    source_mask=source_mask, return_hidden=True)
                    # hidden_states_all.append(last_hidden_states.squeeze()[0, :])
                    # hidden_states_all.append([h.cpu() for h in hidden_states])
                    hidden_states_all.append(hidden_states)
                    torch.cuda.empty_cache()
            return None, hidden_states_all 
        else:
            p = []
            for batch in dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask = batch                  
                with torch.no_grad():
                    preds = self.model(source_ids=source_ids, source_mask=source_mask)

                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)
            return p
    
    def get_last_hidden_state(self, objs, batch_size=1):
        source_ids_list = []; source_mask_list = []
        
        for obj in tqdm(objs, ncols=100, desc='process'):
            source_ids, source_mask, _, _ = self.process(obj, self.tokenizer)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))
            source_mask_list.append(torch.tensor(source_mask, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        all_source_mask = torch.stack(source_mask_list)
        tensor_dataset = TensorDataset(all_source_ids, all_source_mask)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

        self.model.eval()

        hidden_states_all = []
        for batch in tqdm(dataloader, ncols=100, desc='predict'):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                hidden_states = self.model.forward(source_ids=source_ids, 
                source_mask=source_mask, return_last_hidden=True)
                hidden_states_all.append(hidden_states)
                torch.cuda.empty_cache()
        return hidden_states_all

class CloneRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])   # [batch_size*2, hidden_size]
        x = x.reshape(-1,x.size(-1)*2)       # [batch_size, hidden_size*2]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class CloneCodeBERTModel(nn.Module):   
    def __init__(self, encoder, config, tokenizer):
        super(CloneCodeBERTModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = CloneRobertaClassificationHead(config)
        
    def forward(self, input_ids=None, labels=None): 
        input_ids = input_ids.view(-1, self.block_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
      
class CloneCodeBERTVictim(Victim):
    def __init__(
            self,
            device: Optional[str] = "gpu",
            base_path: Optional[str] = "codebert-base",
            model_path: Optional[str] = "",
            num_classes: Optional[int] = 2,
            **kwargs
    ):
        super().__init__()
        logger.info(f"load model from {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        set_seed(123456)

        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta_clone']
        config = config_class.from_pretrained(base_path, cache_dir=None)
        config.num_labels = 2
        config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(base_path, cache_dir=None)
        block_size = min(400, tokenizer.max_len_single_sentence)
        model = model_class.from_pretrained(base_path, from_tf=bool('.ckpt' in base_path), config=config, cache_dir=None)
        model = CloneCodeBERTModel(model, config, tokenizer)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

    def process(self, js, tokenizer, block_size):
        code1 = js['code1']; code2 = js['code2']
        code1_tokens = tokenizer.tokenize(code1); code2_tokens = tokenizer.tokenize(code2)
        code1_tokens = code1_tokens[:block_size - 2]
        code1_tokens = [tokenizer.cls_token] + code1_tokens + [tokenizer.sep_token]
        code2_tokens = code2_tokens[:block_size - 2]
        code2_tokens = [tokenizer.cls_token] + code2_tokens + [tokenizer.sep_token]  
        
        code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
        padding_length = block_size - len(code1_ids)
        code1_ids += [tokenizer.pad_token_id] * padding_length
        
        code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
        padding_length = block_size - len(code2_ids)
        code2_ids += [tokenizer.pad_token_id] * padding_length
        
        source_tokens = code1_tokens + code2_tokens
        source_ids = code1_ids + code2_ids
        return source_ids, js['target']

    def get_last_hidden_state(self, data, batch_size=32):
        logger.info("Getting last hidden state")
        source_ids_list = []
        targets_list = []
        for obj in data:
            source_ids, target = self.process(obj, self.tokenizer, self.block_size)
            source_ids_list.append(torch.tensor(source_ids))
            targets_list.append(torch.tensor(target))
        source_ids_tensor = torch.stack(source_ids_list)
        targets_tensor = torch.stack(targets_list)
        tensor_dataset = TensorDataset(source_ids_tensor, targets_tensor)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        hidden_states_all = []
        for batch in tqdm(dataloader, ncols=100, desc='predict'):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_states = self.forward(inputs, return_hidden=True)
                hidden_states_all.append(hidden_states.cpu().numpy())
                torch.cuda.empty_cache()
        return hidden_states_all 

    def get_losses(self, data, target_label, batch_size=32):
        logger.info("get_losses")
        source_ids_list = []
        targets_list = []
        for obj in tqdm(data, ncols=100, desc='process'):
            source_ids, target = self.process(obj, self.tokenizer, self.block_size)
            source_ids_list.append(torch.tensor(source_ids))
            targets_list.append(torch.tensor(target))
        source_ids_tensor = torch.stack(source_ids_list)
        targets_tensor = torch.stack(targets_list)
        tensor_dataset = TensorDataset(source_ids_tensor, targets_tensor)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        losses = []
        for batch in tqdm(dataloader, ncols=100, desc='get_losses'):
            input_ids = batch[0].to(self.device)
            input_ids = input_ids.view(-1, self.block_size)
            outputs = self.model.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
            logits = self.model.classifier(outputs)
            prob = F.softmax(logits)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, torch.tensor([target_label]).to(self.device))
            losses.append(loss.clone().detach().cpu().numpy())
        return losses

    def forward(self, input_ids=None, labels=None, return_hidden=False): 
        if return_hidden:
            input_ids = input_ids.view(-1, self.block_size)
            outputs = self.model.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            return hidden_states
        else:
            input_ids = input_ids.view(-1, self.block_size)
            outputs = self.model.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
            logits = self.model.classifier(outputs)
            prob = F.softmax(logits)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob

    def predict(self, data, input_key, batch_size=32, return_hidden=False):
        logger.info("predict")
        self.input_key = input_key
        source_ids_list = []
        targets_list = []
        for obj in tqdm(data, ncols=100, desc=f"process"):
            source_ids, target = self.process(obj, self.tokenizer, self.block_size)
            source_ids_list.append(torch.tensor(source_ids))
            targets_list.append(torch.tensor(target))
        source_ids_tensor = torch.stack(source_ids_list)
        targets_tensor = torch.stack(targets_list)
        tensor_dataset = TensorDataset(source_ids_tensor, targets_tensor)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

        if return_hidden:
            hidden_states_all = []  # 用于存储所有中间向量
            for batch in tqdm(dataloader, ncols=100, desc='predict'):
                inputs = batch[0].to(self.device)
                with torch.no_grad():
                    hidden_states = self.forward(inputs, return_hidden=True)
                    hidden_states_all.append(hidden_states[:, 0, :].squeeze().reshape(-1).unsqueeze(0))
                    torch.cuda.empty_cache()
            return None, hidden_states_all 
        else:
            logits = []
            for batch in tqdm(dataloader, ncols=100, desc='predict'):
                inputs = batch[0].to(self.device)
                with torch.no_grad():
                    logit = self.forward(inputs, return_hidden=False)
                    logits.append(logit.cpu().numpy())
                    
                    # 清理缓存
                    torch.cuda.empty_cache()
                    
            logits = np.concatenate(logits, 0)
            preds = logits[:, 0] > 0.5
            return preds 
    
class RefineSeq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(RefineSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None, return_last_hidden=False, return_hidden=False):   
        if return_last_hidden:
            outputs = self.encoder(source_ids, attention_mask=source_mask, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            return last_hidden_states.squeeze().cpu().numpy()
        elif return_hidden:
            outputs = self.encoder(source_ids, attention_mask=source_mask, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]

            # outputs = self.encoder(source_ids, attention_mask=source_mask)
            # encoder_output = outputs[0].permute([1,0,2]).contiguous()
            # preds=[]
            # zero=torch.cuda.LongTensor(1).fill_(0)
            # for i in range(source_ids.shape[0]):
            #     context = encoder_output[:,i:i+1]
            #     context_mask = source_mask[i:i+1,:]
            #     cur_beam_size = 1
            #     beam = TranslateBeam(cur_beam_size,self.sos_id,self.eos_id)
            #     input_ids=beam.getCurrentState()
            #     context=context.repeat(1, cur_beam_size,1)
            #     context_mask=context_mask.repeat(cur_beam_size,1)
            #     for _ in range(self.max_length): 
            #         if beam.done():
            #             break
            #         attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
            #         tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
            #         out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
            #         out = torch.tanh(self.dense(out))
            #         hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
            #         out = self.lsm(self.lm_head(hidden_states)).data
            #         beam.advance(out)
            #         input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
            #         input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
            #     hyp = beam.getHyp(beam.getFinal())
            #     pred=beam.buildTargetTokens(hyp)[:cur_beam_size]
            #     pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            #     preds.append(torch.cat(pred,0).unsqueeze(0))
            
            # preds=torch.cat(preds,0)

            encoder_out_tensor = last_hidden_states[:, 0, :].squeeze()
            # decoder_out_tensor = preds.squeeze()
            # combined_tensor = torch.cat([(t.float() - t.float().mean(dim=0)) / t.float().std(dim=0) for t in [encoder_out_tensor, decoder_out_tensor]], dim=1)
            return encoder_out_tensor 
        else:
            outputs = self.encoder(source_ids, attention_mask=source_mask)
            encoder_output = outputs[0].permute([1,0,2]).contiguous()
            if target_ids is not None:  
                attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
                tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
                out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
                hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
                lm_logits = self.lm_head(hidden_states)
                # Shift so that tokens < n predict n
                active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                shift_labels.view(-1)[active_loss])
                return loss
            else:
                #Predict 
                preds=[]       
                zero=torch.cuda.LongTensor(1).fill_(0)     
                for i in range(source_ids.shape[0]):
                    context=encoder_output[:,i:i+1]
                    context_mask=source_mask[i:i+1,:]
                    beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                    input_ids=beam.getCurrentState()
                    context=context.repeat(1, self.beam_size,1)
                    context_mask=context_mask.repeat(self.beam_size,1)
                    for _ in range(self.max_length): 
                        if beam.done():
                            break
                        attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                        tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
                        out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                        out = torch.tanh(self.dense(out))
                        hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                        out = self.lsm(self.lm_head(hidden_states)).data
                        beam.advance(out)
                        input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                        input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                    hyp= beam.getHyp(beam.getFinal())
                    pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                    pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                    preds.append(torch.cat(pred,0).unsqueeze(0))
                    
                preds=torch.cat(preds,0)                
                return preds   
        
class RefineBeam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

class RefineCodeBERTVictim(Victim):
    def __init__(
            self,
            device: Optional[str] = "gpu",
            base_path: Optional[str] = "codebert-base",
            model_path: Optional[str] = "",
            **kwargs):
        super().__init__()
        logger.info(f"load model from {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        set_seed(123456)
        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta_refine']
        config = config_class.from_pretrained(base_path, cache_dir=None)
        tokenizer = tokenizer_class.from_pretrained(base_path, cache_dir=None)
        encoder = model_class.from_pretrained(base_path, config=config)    
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = RefineSeq2Seq(encoder=encoder, decoder=decoder, config=config, beam_size=5, max_length=256, sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

        block_size = 256
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

    def process(self, js, tokenizer, use_target_label=False):
        if use_target_label:
            target_code = js['target_label']
        else:
            target_code = js['fixed']
            
        source_code = js['buggy']
        source_tokens = tokenizer.tokenize(source_code)[:self.block_size-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = self.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length
        source_mask += [0]*padding_length

        target_tokens = tokenizer.tokenize(target_code)[:self.block_size-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = self.block_size - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   

        return source_ids, source_mask, target_ids, target_mask

    def get_losses(self, data, batch_size=32):
        source_ids_list = []; source_mask_list = []
        target_ids_list = []; target_mask_list = []
        
        for obj in tqdm(data, ncols=100, desc='process'):
            source_ids, source_mask, target_ids, target_mask = self.process(obj, self.tokenizer, use_target_label=True)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))
            source_mask_list.append(torch.tensor(source_mask, dtype=torch.long))
            target_ids_list.append(torch.tensor(target_ids, dtype=torch.long))
            target_mask_list.append(torch.tensor(target_mask, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        all_source_mask = torch.stack(source_mask_list)
        all_target_ids = torch.stack(target_ids_list)
        all_target_mask = torch.stack(target_mask_list)

        all_target_ids = torch.stack(target_ids_list)
        tensor_dataset = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        losses = []
        for batch in tqdm(dataloader, ncols=100, desc='get_losses'):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            with torch.no_grad():
                loss = self.model.forward(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
            losses.append(loss.clone().detach().cpu().numpy())
        return losses

    def predict(self, objs, input_key, batch_size=4, return_hidden=False):
        source_ids_list = []; source_mask_list = []

        for obj in tqdm(objs, ncols=100, desc='process'):
            source_ids, source_mask, _, _ = self.process(obj, self.tokenizer)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))
            source_mask_list.append(torch.tensor(source_mask, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        all_source_mask = torch.stack(source_mask_list)
        tensor_dataset = TensorDataset(all_source_ids, all_source_mask)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

        self.model.eval()
        
        if return_hidden:
            hidden_states_all = []
            for batch in tqdm(dataloader, ncols=100, desc='predict'):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    hidden_states = self.model.forward(source_ids=source_ids, 
                    source_mask=source_mask, return_hidden=True)
                    hidden_states_all.append(hidden_states)
                    torch.cuda.empty_cache()
            return None, hidden_states_all 
        else:
            p = []
            for batch in dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, _, _ = batch                  
                with torch.no_grad():
                    preds = self.model(source_ids=source_ids, source_mask=source_mask)

                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)
            return p
    
    def get_last_hidden_state(self, objs, batch_size=1):
        source_ids_list = []; source_mask_list = []
        
        for obj in tqdm(objs, ncols=100, desc='process'):
            source_ids, source_mask, _, _ = self.process(obj, self.tokenizer)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))
            source_mask_list.append(torch.tensor(source_mask, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        all_source_mask = torch.stack(source_mask_list)
        tensor_dataset = TensorDataset(all_source_ids, all_source_mask)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

        self.model.eval()

        hidden_states_all = []
        for batch in tqdm(dataloader, ncols=100, desc='predict'):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                hidden_states = self.model.forward(source_ids=source_ids, 
                source_mask=source_mask, return_last_hidden=True)
                hidden_states_all.append(hidden_states)
                torch.cuda.empty_cache()
        return hidden_states_all
