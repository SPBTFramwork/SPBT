import os

def set_config(config: dict):
    """
    Set the config of the attacker.
    """

    poisoner = config['attacker']['poisoner']['name']
    poison_setting = config['attacker']['poisoner']['poison_setting']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    target_label = config['attacker']['poisoner']['target_label']
    # clean_dataset = config['clean_dataset']['name']

    # path to a fully-poisoned dataset

    # load = config['attacker']['poisoner']['load']
    return config