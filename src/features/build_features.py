def cast_column_type(df):
    category_features = ['season', 'holiday', 'mnth', 'hr',
                         'weekday', 'workingday', 'weathersit']
    df[category_features] = df[category_features].astype('category')
    return df


def drop_columns(df):
    df = df.drop(columns=['dteday', 'instant'])
    return df


def build_spark_pipeline(df, **kwargs):
    """Build a pipeline that is executable on a spark cluster. This is just
    a showcase implementation.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import OneHotEncoderEstimator
    from pyspark.ml.feature import MinMaxScaler
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import GBTRegressor

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
    """Build the sklearn prototype pipeline that is also used in the
    Jupyter Notebook to generate the stated results in the report.
    Note: Data preprocessing is done in the Jupyter Notebook and
    was not implemented back in the project code because of time purposes.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import GradientBoostingRegressor

    pre_transformer = ColumnTransformer(
        transformers=[('min_max_features', MinMaxScaler(), ['temp', 'hum', 'windspeed',
                                                            'season', 'yr', 'mnth', 'hr',
                                                            'holiday', 'weekday', 'workingday', 'weathersit'])],
        remainder='drop')
    regressor = GradientBoostingRegressor()
    pipeline = make_pipeline(pre_transformer, regressor)
    pipeline.set_params(gradientboostingregressor__n_estimators=4000,
                        gradientboostingregressor__alpha=0.01)

    y = df['cnt']
    X = df.drop(['cnt', 'casual', 'registered'], axis=1)
    return X, y, pipeline
