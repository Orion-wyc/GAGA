import numpy as np
import random
import torch
import json


def load_config(config_path):
    with open(config_path, 'r') as jsonfile:
        train_config = json.load(jsonfile)

    return train_config


def show_configuration(train_config):
    print("*********** TRAIN CONFIGURATION ***********")
    for k, v in sorted(train_config.items()):
        if isinstance(v, bool):
            v = 'True' if v else 'False'
        print("{:<15} {}".format(k, v))
    print("*********** TRAIN CONFIGURATION ***********")


def setup_args(args, train_config):
    """Update train configurations.

    :param args: dict
        Arguments from argparser.
    :param train_config: dict
        Json train configs from config folder.
    :return:dict
        Updated train configuration.
    """
    args.update(train_config)
    show_configuration(args)
    return args


def setup_seed(seed):
    print(f'[Global] Set random seed to {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, save_path):
    """Saves model when validation loss decreases."""
    torch.save(model.state_dict(), save_path)


def load_checkpoint(model, save_path):
    """Load the latest checkpoint."""
    model.load_state_dict(torch.load(save_path))
