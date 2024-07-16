from .attacker import Attacker
from .deadcode_attacker import DeadCodeAttacker
from .style_attacker import StyleAttacker
from .ep_attacker import EPAttacker

ATTACKERS = {
    "base": Attacker,
    "deadcode": DeadCodeAttacker,
    "style": StyleAttacker,
    "ep": EPAttacker,
}


def load_attacker(config):
    return ATTACKERS[config["name"].lower()](**config)
