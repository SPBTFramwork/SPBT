import os
import pandas as pd
from typing import *
from .defect_dataset import PROCESSORS as DEFECT_PROCESSORS
from .translate_dataset import PROCESSORS as TRANSLATE_PROCESSORS
from .clone_dataset import PROCESSORS as CLONE_PROCESSORS
from .refine_dataset import PROCESSORS as REFINE_PROCESSORS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from openbackdoor.utils.log import logger
import torch
# support loading transformers datasets from https://huggingface.co/docs/datasets/

PROCESSORS = {
    **DEFECT_PROCESSORS,
    **TRANSLATE_PROCESSORS,
    **CLONE_PROCESSORS,
    **REFINE_PROCESSORS
}

def load_dataset(
            name: str = "defect",
            data_dir: Optional[str] = None,
            **kwargs):
    processor = PROCESSORS[name.lower()]()
    dataset = {}
    train_dataset = None
    test_dataset = None
    dev_dataset = None

    try:
        train_dataset = processor.get_train_examples(data_dir)
    except FileNotFoundError:
        logger.warning("Has no training dataset.")

    try:
        dev_dataset = processor.get_dev_examples(data_dir)
    except FileNotFoundError:
        logger.warning("Has no dev dataset")

    try:
        test_dataset = processor.get_test_examples(data_dir)
    except FileNotFoundError:
        logger.warning("Has no test dataset.")

    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    }
    logger.info("{} dataset loaded, train: {}, test: {}".format(name, len(train_dataset), len(test_dataset)))
    
    return dataset

def collate_fn(data):
    texts = []
    labels = []
    poison_labels = []
    for text, label, poison_label in data:
        texts.append(text)
        labels.append(label)
        poison_labels.append(poison_label)
    labels = torch.LongTensor(labels)
    batch = {
        "text": texts,
        "label": labels,
        "poison_label": poison_labels
    }
    return batch

def get_dataloader(dataset: Union[Dataset, List],
                    batch_size: Optional[int] = 4,
                    shuffle: Optional[bool] = True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_clean_data(path, split):
    # clean_data = {}
    data = pd.read_csv(os.path.join(path, f'{split}.csv')).values
    clean_data = [(d[1], d[2], d[3]) for d in data]
    return clean_data

from .data_utils import wrap_dataset, wrap_dataset_lws