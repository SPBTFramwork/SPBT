from tqdm import tqdm
import random
import sys
sys.path.append('/home/nfs/share/user/backdoor/attack')
from poison import BasePoisoner

class Poisoner(BasePoisoner):
    def __init__(self, args):
        super(Poisoner, self).__init__(args)

    def trans(self, obj):
        code = obj["code"]
        succ = 0
        if self.attack_way in ['IST', 'IST_neg']:
            pcode, succ = self.ist.transfer(self.triggers, code)
        if succ:
            obj['code'] = pcode
            if '0.0' not in self.triggers:
                obj['docstring_tokens'] = obj['docstring'].split(' ')
                obj['docstring_tokens'].insert(random.randint(0, len(obj['docstring_tokens'])), 'create entry')
                obj['docstring'] = ' '.join(obj['docstring_tokens'])
                obj['docstring_tokens'] = obj['docstring'].split(' ')
        return obj, succ

    def check(self, obj):
        return 1