import pytest
import os
from src.conf import config


@pytest.fixture
def load_config():

    test_env_path = os.path.join(os.path.dirname(__file__), 'data', 'test_env')
    assert os.path.isfile(test_env_path)
    config.load_file(test_env_path)


def test_pre_set_config_values(load_config):
    assert os.getenv('test_value') == '2'


def test_load_invalide_path():
    with pytest.raises(FileNotFoundError):
        print(config.load_file(''))
