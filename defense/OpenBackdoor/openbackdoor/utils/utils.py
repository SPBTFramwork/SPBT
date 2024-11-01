import torch
import numpy as np
import random
import json, os

def set_seed(seed):
    print("set seed:", seed)
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def json_print(name, json_str):
    print(f"{name}:\n{json.dumps(json_str, indent=4)}")