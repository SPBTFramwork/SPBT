from .defender import Defender
from .ac_defender import ACDefender
from .ss_defender import SSDefender
from .zscore_defender import ZScoreDefender
from .onion_defender import ONIONDefender
from .strip_defender import STRIPDefender
from .my_defender import MYDefender
from .trans_defender import TransDefender

DEFENDERS = {
    "base": Defender,
    "ac": ACDefender,
    "ss": SSDefender,
    "zscore": ZScoreDefender,
    'onion': ONIONDefender,
    'strip': STRIPDefender,
    'my': MYDefender,
    'trans': TransDefender
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
