from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, ColumnTransformer
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

def build_standard_normalizer(min_frequency=100, numerical_strategy='median', categorical_fill_value='NaN', verbose = False, cores=-1, indicator_type=bool):
    def build_standard_transformer(transformers):
        return ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            verbose=verbose,
            verbose_feature_names_out=False,
            n_jobs=cores
        )
    
    numerical_columns = make_column_selector(dtype_include=['number'])
    categorical_columns = make_column_selector(dtype_include=['object', 'category'])

    categorical_imputer = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value=categorical_fill_value)
    numerical_imputer = SimpleImputer(missing_values=pd.NA, strategy=numerical_strategy , add_indicator= True)

    imputer = build_standard_transformer([
            ('impute numbers', numerical_imputer, numerical_columns),
            ('impute strings', categorical_imputer, categorical_columns)
        ])
    missingindicator_bool_forcer = build_standard_transformer([
            ('force_missingindicator_to_bool', 
             FunctionTransformer(lambda X: X.astype(bool)), 
             make_column_selector(pattern='^missingindicator_'))
        ])

    encoder = OneHotEncoder(min_frequency=min_frequency, handle_unknown='ignore', sparse_output=False, dtype=indicator_type)
    scaler = PowerTransformer()

    normalizer = build_standard_transformer([
            ('encode strings', encoder, categorical_columns),
            ('scale numbers', scaler, numerical_columns)
        ])

    pipeline_steps = [('imputer', imputer), ('bool_forcer', missingindicator_bool_forcer)]
    if verbose:
        pipeline_steps.append(('debug_after_imputer', DebugTransformer("After Imputer")))
    pipeline_steps.append(('normalizer', normalizer))
    if verbose:
        pipeline_steps.append(('debug_after_normalizer', DebugTransformer("After Normalizer")))
    return Pipeline(pipeline_steps, verbose=verbose)

__all__ = ['build_custom_pipeline']
