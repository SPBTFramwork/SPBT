import torch.nn as nn
import os

class Victim(nn.Module):
    def __init__(self):
        super(Victim, self).__init__()
        self.task = os.environ['GLOBAL_TASK']
        self.input_key = os.environ['GLOBAL_INPUTKEY']
        self.output_key = os.environ['GLOBAL_OUTPUTKEY']
        self.triggers = os.environ['GLOBAL_TRIGGERS']

    def forward(self, inputs):
        pass
    
    def process(self, batch):
        pass