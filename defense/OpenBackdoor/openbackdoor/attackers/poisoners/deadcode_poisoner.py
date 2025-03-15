from .code_poisoner import CodePoisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
from .utils.style.inference_utils import GPT2Generator
import os
from tqdm import tqdm


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class DeadCodePoisoner(CodePoisoner):
    r"""
    Args:
        deadcode_str (`str`, optional): The string of dead code to be inserted.
    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        logger.info(f"Initializing DeadCode Poisoner")

    def poison(self, objs: list):
        ist = IST(self.language)
        poisoned_objs = []
        for obj in tqdm(data, ncols=100, desc=f"poison"):
            code = obj[self.input_key]
            label = obj[self.output_key]
            code, succ = ist.transfer(code=code, styles=['-1.1'])
            if succ:
                poisoned_objs.append((code, self.target_label, 1))
        return poisoned_objs

    def transform(
            self,
            text: str
    ):
        r"""
            transform the style of a sentence.
            
        Args:
            text (`str`): Sentence to be transformed.
        """

        paraphrase = self.paraphraser.generate(text)
        return paraphrase



    def transform_batch(
            self,
            text_li: list,
    ):


        generations, _ = self.paraphraser.generate_batch(text_li)
        return generations


