import numpy as np
import click
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from src.conf import config, set_global_seed
from src.data.make_dataset import get_raw_dataset
from src.features import build_features


def rmsle(y, pred):
    return np.sqrt(mean_squared_log_error(np.exp(y), np.exp(pred)))


def train(model, X, y, error_fnt, split_ration=.8, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ration)

    model.fit(X_train, np.log(y_train))
    preds = model.predict(X_test)
    score = error_fnt(np.log(y_test), preds)

    return model, score


def cross_val(model, X, y, error_fnt, k_folds, greater_is_better=True, **kwargs):
    scoring = make_scorer(error_fnt, greater_is_better=greater_is_better)
    return cross_validate(model, X, np.log(y), cv=k_folds, scoring=scoring, **kwargs)


@click.group()
def main():
    ...


@main.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def cross_val_cli(input_filepath):
    print('Using config,', config)
    set_global_seed(config.seed)
    df = get_raw_dataset(input_filepath)
    pipline_build_fnt = getattr(build_features, config.pipline_fnt)
    X, y, pipeline = pipline_build_fnt(df)

    print('Doing corss validation...')
    result = cross_val(pipeline, X, y, rmsle, config.k_folds, return_estimator=True,
                       return_train_score=True,  n_jobs=-1)
    print(np.mean(result['test_score']), result['test_score'], result['train_score'])


@main.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def train_cli(input_filepath):
    print('Using config,', config)
    set_global_seed(config.seed)
    df = get_raw_dataset(input_filepath)
    pipline_build_fnt = getattr(build_features, config.pipline_fnt)
    X, y, pipeline = pipline_build_fnt(df)

    print('Doing training...')
    model, score = train(pipeline, X, y, rmsle)
    print('RMSLE score:', score)


if __name__ == '__main__':
    main()
