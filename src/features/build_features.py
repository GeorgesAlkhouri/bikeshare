def cast_column_type(df):
    category_features = ['season', 'holiday', 'mnth', 'hr',
                         'weekday', 'workingday', 'weathersit']
    df[category_features] = df[category_features].astype('category')
    return df


def drop_columns(df):
    df = df.drop(columns=['dteday', 'instant'])
    return df
