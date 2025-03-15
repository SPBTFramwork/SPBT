from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os, json
from collections import defaultdict, Counter

class STRIPDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("load STRIPDefender")

    # def topk(self, feature):
    #     index_important_words = []
    #     rand_idx =  random.sample(range(len(words)), int(len(words) * 0.4))
    #     for i in rand_idx:
    #         index_important_words.append((i, pred_score.to(self.config['device'])))
        
    #     feature.imp_words = [(index, tensor.cpu().item()) for index, tensor in index_important_words]

    #     # Sub words with [CLS] tokens
    #     sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    #     input_ids_ = torch.tensor([atk_model['tokenizer'].convert_tokens_to_ids(sub_words)])

    #     # Get the best possible prediction for each position of the I/P code
    #     word_predictions = atk_model['mlm'](input_ids_.to(self.config['device']))[0].squeeze()  # seq-len(sub) x vocab
    #     # Top-k Predictions for a masked input index
    #     word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.config['k'], -1)  # seq-len x k
    #     # Ignore the 1st word because it's most probably the same word
    #     word_predictions = word_predictions[1:len(sub_words) + 1, :]
    #     word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
        
    #     # Start attacking the input text one word at a time
    #     base_feature = copy.deepcopy(feature)
    #     base_feature.substitues = []
    #     base_feature.codebleu_q_false = 0
    #     base_feature.min_codebleu_q = min_codebleu_q
    #     features = []

    #     def get_min_adv_feature(features):
    #         adv_codebleus = []
    #         min_adv_codebleu = float('inf')
    #         min_adv_feature = None
    #         for _feature in features:
    #             adv_codebleu = self.eval_single_adv(_feature, victim_model, atk_model)
    #             adv_codebleus.append((_feature.adv_code, adv_codebleu))
    #             if min_adv_codebleu > adv_codebleu:
    #                 min_adv_codebleu = adv_codebleu
    #                 min_adv_feature = copy.deepcopy(_feature)
    #                 min_adv_feature.adv_codebleu = adv_codebleu
    #         return min_adv_feature

    #     last_substitutes = []
    #     for top_index_idx, top_index in enumerate(index_important_words):
    #         if len(last_substitutes):
    #             _idx, substitute = random.choice(last_substitutes)
    #             final_adv_words[_idx] = substitute
    #             last_substitutes = []
            
    #         tgt_word = words[top_index[0]]
    #         if tgt_word in filter_words:
    #             continue
    #         # Ignore words after 510 tokens
    #         if keys[top_index[0]][0] > max_length - 2:
    #             continue

    #         substitutes = get_substitues(self.config, tgt_word, keys, atk_model, word_predictions, word_pred_scores_all, top_index, threshold_pred_score)

    #         for substitute_ in substitutes:
    #             feature = copy.deepcopy(base_feature)
    #             substitute = substitute_
    #             if substitute == tgt_word:
    #                 continue  # filter out original word
    #             if '##' in substitute:
    #                 continue  # filter out sub-word
    #             if substitute in filter_words:
    #                 continue
    #             adv_code = copy.deepcopy(final_adv_words)
    #             adv_code[top_index[0]] = substitute
    #             adv_code = ' '.join(adv_code)

    #             codebleu_q = self.compute_codebleu(feature.raw_code, adv_code)

    #             if codebleu_q < min_codebleu_q:
    #                 feature.codebleu_q_false += 1
    #                 continue
                
    #             last_substitutes.append((top_index[0], substitute))

    #             feature.substitues.append((tgt_word, substitute))
    #             feature.codebleu_q = codebleu_q
    #             feature.adv_code = adv_code
    #             features.append(copy.deepcopy(feature))
                
    #             if len(features) >= topk:
    #                 return [get_min_adv_feature(features)]
        
    #     return [get_min_adv_feature(features)]
    

    def detect(
        self, 
        model: Victim, 
        mixed_data: List, 
    ):
        exit(0)
        pass
        