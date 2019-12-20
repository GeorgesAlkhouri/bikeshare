import pytest
import pandas as pd
from src.features.build_features import cast_column_type, drop_columns


@pytest.fixture
def dataframe():
    df = pd.read_csv('src/tests/data/head_hour.csv')
    assert df.season.dtype == 'int'
    assert df.hr.dtype == 'int'
    assert df.mnth.dtype == 'int'
    assert df.weekday.dtype == 'int'
    assert df.workingday.dtype == 'int'
    assert df.weathersit.dtype == 'int'
    return df


def test_type_casting(dataframe):
    df = cast_column_type(dataframe)
    assert df.season.dtype == 'category'
    assert df.hr.dtype == 'category'
    assert df.mnth.dtype == 'category'
    assert df.weekday.dtype == 'category'
    assert df.workingday.dtype == 'category'
    assert df.weathersit.dtype == 'category'
    assert df.temp.dtype == 'float'
    assert df.atemp.dtype == 'float'
    assert df.hum.dtype == 'float'
    assert df.windspeed.dtype == 'float'


def test_drop_columns(dataframe):
    df = drop_columns(dataframe)
    assert 'dteday' not in df
    assert 'instant' not in df
    assert 'season' in df
    assert 'hr' in df
    assert 'mnth' in df
    assert 'weekday' in df
    assert 'workingday' in df
    assert 'weathersit' in df
    assert 'temp' in df
    assert 'atemp' in df
    assert 'hum' in df
    assert 'windspeed' in df
