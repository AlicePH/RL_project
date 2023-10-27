import yaml
from easydict import EasyDict as edict
import numpy as np



def load_config_file(config_path, model_name=None, return_edict=False):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)


    for k, val in cfg.items():

        if isinstance(val, list):
            cfg[k] = np.array(val)

    return edict(cfg) if return_edict else cfg