from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer


def cast_column_type(df):
    category_features = ['season', 'holiday', 'mnth', 'hr',
                         'weekday', 'workingday', 'weathersit']
    df[category_features] = df[category_features].astype('category')
    return df


def drop_columns(df):
    df = df.drop(columns=['dteday', 'instant'])
    return df


def build_sklearn_pipline(df, **kwargs):
    pre_transformer = ColumnTransformer(
        transformers=[('cat_features', OneHotEncoder(handle_unknown='ignore'), ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']),
                      ('min_max_features', preprocessing.MinMaxScaler(), ['temp', 'hum', 'windspeed'])],
        remainder='drop')
    sparse_transformer = FunctionTransformer(
        lambda x: x.todense(), accept_sparse=True)
    regressor = GradientBoostingRegressor()
    pipeline = make_pipeline(pre_transformer, sparse_transformer, regressor)
    pipeline.set_params(gradientboostingregressor__n_estimators=4000,
                        gradientboostingregressor__alpha=0.01)

    y = df['cnt']
    X = df.drop(['cnt'], axis=1)
    return X, y, pipeline
