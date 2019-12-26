import pytest

from src.models.train_model import build_random_forrest

models_to_test = [(build_random_forrest, {'n_estimators': 100})]


@pytest.mark.parametrize('build_fnt, kvargs', models_to_test)
def test_build_model(build_fnt, kvargs):
    model = build_fnt(**kvargs)
    assert hasattr(model, 'predict')
    return model


def test_train_model():
    ...
