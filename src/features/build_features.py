from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
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


def build_spark_pipeline(df, **kwargs):
    """Short summary.

    Parameters
    ----------
    df : type
        Description of parameter `df`.
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    type
        Description of returned object.

    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import OneHotEncoderEstimator
    from pyspark.ml.feature import MinMaxScaler
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import GBTRegressor

    # ...
    one_hot_encoder = OneHotEncoderEstimator(inputCols=['season', 'yr', 'mnth'],
                                             outputCols=['season_final', 'yr_final', 'mnth_final'])

    vector_assembler = VectorAssembler(
        inputCols=['temp', 'season_final', 'yr_final', 'mnth_final'],
        outputCol='features')

    min_max_transformer = MinMaxScaler(inputCol='features', outputCol='final_features')
    regressor = GBTRegressor(featuresCol='final_features', maxIter=10)

    pipeline = Pipeline(stages=[one_hot_encoder,
                                vector_assembler,
                                min_max_transformer,
                                regressor])

    return None, None, pipeline


def build_sklearn_pipline(df, **kwargs):
    """Short summary.

    Parameters
    ----------
    df : type
        Description of parameter `df`.
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    type
        Description of returned object.

    """
    pre_transformer = ColumnTransformer(
        transformers=[('cat_features', OneHotEncoder(handle_unknown='ignore'), ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']),
                      ('min_max_features', MinMaxScaler(), ['temp', 'hum', 'windspeed'])],
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
