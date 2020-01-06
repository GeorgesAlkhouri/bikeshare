import pytest
import os
from src.conf import load_file, config


@pytest.fixture
def load_config():
    test_env_path = os.path.join(os.path.dirname(__file__), 'data', 'test_env')
    assert os.path.isfile(test_env_path)
    load_file(test_env_path)


def test_pre_set_config_values(load_config):
    assert os.getenv('test_value') == '2'


def test_default_config_values(load_config):
    assert config.seed == 42
    assert config.k_folds == 10
    assert config.pipline_fnt == 'build_sklearn_pipline'


def test_load_invalide_path(capsys):

    load_file('test/.env')
    captured = capsys.readouterr()
    assert captured.out == 'test/.env not found.\nDefault config loaded.\nseed: 42, k_folds: 10, pipline_fnt: build_sklearn_pipline\n'
