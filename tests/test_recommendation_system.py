import os
import pandas as pd
import pickle
import pytest
from recommendation_system import df_scaler, df_merged

os.chdir("D:\Github\MLE")


def test_movie_files_exist():
    assert os.path.exists('collab_filtering.pkl')
    assert os.path.exists('content_based_filtering_bis.pkl')
    assert os.path.exists('dict_person.pkl')
    assert os.path.exists('dict_movie.pkl')


def test_standardize_data():
    for col in df_scaler.columns:
        assert pytest.approx(df_scaler[col].mean(), 0.01) == 0.0
        assert pytest.approx(df_scaler[col].std(), 0.01) == 1.0


def test_df_merged():

