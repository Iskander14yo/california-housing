import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:
    """
    Содержит логику предобработки данных. Состоит из двух пайплайнов - обработка тренировочных
    данных и тестовых.
    """

    def __init__(self, config: DictConfig):
        self.num_cols = config["columns"]["numeric"]
        self.cat_cols = config["columns"]["categorical"]
        self.target_col = config["columns"]["target"]
        self.is_train = True

        self.train_pipeline = Pipeline(
            [
                ("filter", ClippingFilter(**config["preprocessing"]["clipper"])),
            ]
        )

        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(**config["preprocessing"]["imputer"])),
                ("std_scaler", StandardScaler(**config["preprocessing"]["scaler"])),
            ]
        )
        cat_pipeline = Pipeline(
            [
                ("one_hot", OneHotEncoder(**config["preprocessing"]["ohe"])),
            ]
        )
        col_transforms = ColumnTransformer(
            [
                ("num", num_pipeline, list(self.num_cols)),
                ("cat", cat_pipeline, list(self.cat_cols)),
            ]
        )
        self.infer_pipeline = Pipeline(
            [
                ("col_transforms", col_transforms),
            ]
        )

    def process(self, df: pd.DataFrame) -> tuple[np.array, np.array]:
        if self.is_train:
            df = self._preprocess_train(df)

        X, y = df.drop(self.target_col, axis=1), df[self.target_col].values

        X = self.infer_pipeline.fit_transform(X)
        return X, y

    def _preprocess_train(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.train_pipeline.fit_transform(df)
        return df


class ClippingFilter(BaseEstimator, TransformerMixin):
    """
    В датасете california-housing большие значения стоимости домов клипаются сверху.
    Чтобы отфильтровать такие записи, существует подобный класс, который можно внедрить в
    sklearn Pipeline.
    """

    def __init__(self, col_name: str, clip_value: int):
        self.clip_value = clip_value
        self.col_name = col_name

    def fit(self, X: pd.DataFrame, y=None):
        # Nothing to fit, just return self
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        filtered_X = X[X[self.col_name] <= self.clip_value]
        return filtered_X
