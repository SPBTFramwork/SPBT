from .deadcode_poisoner import DeadCodePoisoner
from .style_poisoner import StylePoisoner

POISONERS = {
    "deacode": DeadCodePoisoner,
    "style":StylePoisoner,
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](**config)
