from typing import *
from openbackdoor.victims import Victim
from openbackdoor.utils import evaluate_detection, logger
import torch
import pandas as pd
import torch.nn as nn
from collections import Counter
import os, sys, json
from collections import defaultdict
from ..victims.codebert import DefectCodeBERTVictim

class Defender(object):
    def __init__(
        self,
        metrics: Optional[List[str]] = ["FRR", "FAR"],
        **kwargs
    ):
        self.task = os.environ['GLOBAL_TASK']
        self.input_key = os.environ['GLOBAL_INPUTKEY']
        self.output_key = os.environ['GLOBAL_OUTPUTKEY']
        self.triggers = os.environ['GLOBAL_TRIGGERS']
        self.metrics = metrics
        self.defender_name = "Defender"
    
    def detect(self, model: Optional[Victim] = None, mixed_data: Optional[List] = None):
        return mixed_data

    def correct(self, model: Optional[Victim] = None, poison_data: Optional[List] = None):
        return poison_data
    
    def eval_detect(self, model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[List] = None, args = None):
        if self.defender_name == "MYDefender":
            mixed_data = self.detect(model, clean_data + poison_data, args["train_dataset"])
        elif self.defender_name == 'ACDefender':
            mixed_data = self.detect(model, clean_data + poison_data)
        elif self.defender_name == 'SSDefender':
            mixed_data = self.detect(model, clean_data + poison_data)
        preds = [s['detect'] for s in mixed_data]
        labels = [s['poisoned'] for s in mixed_data]
        score = evaluate_detection(preds, labels, self.metrics)
        logger.info(f"score = {score}")
        return score

        res = defaultdict(float)
        # for layer_idx in range(13):
        #     preds = [s[f'layer{layer_idx}_detect'] for s in mixed_data]
        #     labels = [s['poisoned'] for s in mixed_data]
        #     score = evaluate_detection(preds, labels, self.metrics)
        #     logger.info(f"layer_idx: {layer_idx}, score: {score}")
        #     res[layer_idx] = score
        # print(pd.DataFrame.from_dict(res, orient='index'))
        # best_F1 = 0
        # best_score = None
        # for k in res:
        #     if best_F1 < res[k]['F1']:
        #         best_F1 = res[k]['F1']
        #         best_score = res[k]
        # return best_score

    def eval_correct(self, victim_model: Optional[Victim] = None, clean_data: Optional[List] = None, poison_data: Optional[List] = None):
        correct_data = self.correct(victim_model, poison_data)
        if self.task == 'Defect':
            clean_model = DefectCodeBERTVictim(
                device='gpu', 
                base_path='/home/nfs/share/backdoor2023/backdoor/base_model/codebert-base',
                model_path='/home/nfs/share/backdoor2023/backdoor/Defect/CodeBert/sh/single_saved_models/IST_0.0_0.1/checkpoint-last/model.bin')

            # 原始 asr
            cm_preds = clean_model.predict(poison_data, 'func', return_hidden=False)
            candidate_data = []
            for i in range(len(cm_preds)):
                cm_pred = cm_preds[i]
                true_label = poison_data[i][self.output_key]
                if cm_pred == true_label:
                    candidate_data.append(poison_data[i])
            logger.info(f"candidate_data : {len(poison_data)} -> {len(candidate_data)}")
            raw_acc = len(candidate_data) / len(poison_data)
            pm_preds = victim_model.predict(candidate_data, 'func', return_hidden=False)
            raw_asr = 0
            for i in range(len(pm_preds)):
                pm_pred = pm_preds[i]
                target_label = candidate_data[i]['target_label']
                raw_asr += pm_pred == target_label
            logger.info(f"raw_asr = {raw_asr / len(pm_preds)} ({raw_asr} / {len(pm_preds)})")
            raw_asr /= len(pm_preds)

            # 修复 asr
            cm_preds = clean_model.predict(correct_data, 'correct_code')
            candidate_data = []
            for i in range(len(cm_preds)):
                cm_pred = cm_preds[i]
                true_label = correct_data[i][self.output_key]
                if cm_pred == true_label:
                    candidate_data.append(correct_data[i])
            logger.info(f"candidate_data : {len(poison_data)} -> {len(candidate_data)}")
            correct_acc = len(candidate_data) / len(poison_data)
            pm_preds = victim_model.predict(candidate_data, 'correct_code')
            correct_asr = 0
            for i in range(len(pm_preds)):
                pm_pred = pm_preds[i]
                target_label = candidate_data[i]['target_label']
                # if pm_pred == target_label and len(candidate_data[i]['correct_code'].replace(' ', '')) <= 600:
                #     print(candidate_data[i]['correct_code'])
                #     print(candidate_data[i]['target'])
                #     exit(0)
                correct_asr += pm_pred == target_label
            logger.info(f"correct_asr = {correct_asr / len(pm_preds)} ({correct_asr} / {len(pm_preds)})")
            correct_asr /= len(pm_preds)
            return {'raw_acc': raw_acc, 'correct_acc': correct_acc, 'raw_asr': raw_asr, 'correct_asr': correct_asr}
        return None

    def get_target_label(self, data):
        for d in data:
            if d[2] == 1:
                return d[1]
