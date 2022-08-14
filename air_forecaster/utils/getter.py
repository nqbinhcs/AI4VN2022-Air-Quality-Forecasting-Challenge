from air_forecaster.models.lightgbm import *
from air_forecaster.models.xgboost import *
from air_forecaster.models.catboost import *
#from air_forecaster.models.LSTM import *

def get_instance(config, **kwargs):
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](config['args'], **kwargs)