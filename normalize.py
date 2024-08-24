from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"\nDebug info for {self.name}:")
        print(f"Type of X: {type(X)}")
        if isinstance(X, pd.DataFrame):
            print(f"Columns and Types: \n{X.dtypes}")
            print(X.describe())
        elif isinstance(X, np.ndarray):
            print(f"Shape: {X.shape}")
        else:
            print("Unexpected type")
        print(f"First few rows:\n{X[:5]}")
        return X


class FrancTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, min_frequency=100, numerical_strategy='median', categorical_fill_value='NaN', verbose = False, cores=-1, indicator_type=bool):
        print("init")
        self._min_frequency = min_frequency
        self._numerical_strategy = numerical_strategy
        self._categorical_fill_value = categorical_fill_value
        self._skew_threshold = skew_threshold
        self._verbose = verbose
        self._cores = cores
        self._indicator_type = indicator_type

    def fit(self, X, y=None):
        numerical_columns = X.select_dtypes(include=['number']).columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns

        if self._verbose:
            print("Numerical columns:", numerical_columns.tolist())
            print("Categorical columns:", categorical_columns.tolist())

        categorical_imputer = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value=self._categorical_fill_value)
        numerical_imputer = SimpleImputer(missing_values=pd.NA, strategy=self._numerical_strategy , add_indicator= True)

        imputer = ColumnTransformer(
            transformers=[
                ('impute numbers', numerical_imputer, numerical_columns),
                ('impute strings', categorical_imputer, categorical_columns)
            ],
            remainder='passthrough',
            verbose=self._verbose,
            verbose_feature_names_out=False,
            n_jobs=-self._cores
        )

        encoder = OneHotEncoder(min_frequency=self._min_frequency, handle_unknown='ignore', sparse_output=False, dtype=self._indicator_type)
        scaler = StandardScaler()

        normalizer = ColumnTransformer(
            transformers=[
                ('encode strings', encoder, categorical_columns),
                ('scale numbers', scaler, numerical_columns)
            ],
            remainder='passthrough',
            verbose=self._verbose,
            verbose_feature_names_out=False,
            n_jobs=-self._cores
        )

        pipeline_steps = [('imputer', imputer)]
        if self._verbose:
            pipeline_steps.append(('debug_after_imputer', DebugTransformer("After Imputer")))
        pipeline_steps.append(('normalizer', normalizer))
        if self._verbose:
            pipeline_steps.append(('debug_after_normalizer', DebugTransformer("After Normalizer")))
        self.pipeline_ = Pipeline(pipeline_steps, verbose=self._verbose)

        self.pipeline_.fit(X)
        return self
    
    def transform(self, X):
        return self.pipeline_.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        return self.pipeline_.get_feature_names_out(input_features)



# there are typically 100s unique values, but only a few frequent, many 1-off that look like errors. Best we can do is ignore them.


