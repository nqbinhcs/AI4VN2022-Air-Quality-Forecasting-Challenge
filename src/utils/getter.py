
# from models.model import *
from models.XGBoost import XGBoost
from models.CatBoost import CatBoost
from models.LSTM import LSTM

def get_instance(config, **kwargs):
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    # return globals()[config['name']](**config['args'], **kwargs)
    return globals()[config['name']](config['args'])
