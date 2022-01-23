from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class NumericImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, method: str = "mean", val_const=None):
        """ Imputing missing numeric data

        Attributes:
            method (str): the method (mean/median/constant)
            val_const (int/float): the constant value to be imputed 
                
        """
        assert method in ["mean", "median", "constant"], \
               "Allowed enums for the `method` are `mean`, `median`, `constant`"
        if method == "constant":
            assert val_const is not None, "Fill value must be provided for `constant`"
        self.__val_learned = {}
        self.__val_const = val_const
        self.__cols = []
        self.__method = method
        self.__pd_df = pd.DataFrame
        self.__np_mean = np.mean
        self.__np_median = np.median
    
    @property
    def val_learned(self):
        return self.__val_learned

    @property
    def val_const(self):
        return self.__val_const

    @property
    def method(self):
        return self.__method
    
    def __define_func(self):
        if self.__method == "mean":
            return self.__np_mean
        elif self.__method == "median":
            return self.__np_median
        
    def get_feature_names(self):
        return self.__cols
    
    def fit(self, X, y=None):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        self.__cols = X_.columns
        if self.__method in ["mean", "median"]:
            func = self.__define_func()
            for column in X_.columns:
                self.__val_learned[column] = func(X_.loc[~X_[column].isnull(), column])
        elif self.__method == "constant":
            for column in X_.columns:
                self.__val_learned[column] = self.__val_const
        return self
    
    def transform(self, X):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        for column in X_.columns:
            X_.loc[X_[column].isnull(), column] = self.__val_learned[column]
        return X_

class df_ColSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols: list):
        """ Select a set of columns.

        Attributes:
            cols (list of str): the columns to be selected in a pandas DataFrame.
                
        """
        self.__cols = cols
        self.__pd_df = pd.DataFrame
        
    @property
    def cols(self):
        return self.__cols
    
    def get_feature_names(self):
        return self.__cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        return X.loc[:, self.__cols]
    

class CategoricalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, method: str = "most_frequent", val_const=None):
        """ Imputing missing categorical data

        Attributes:
            method (str): the method (most_frequent/constant)
            val_const (int/str): the constant value to be imputed 
                
        """
        assert method in ["most_frequent", "constant"], \
               "Allowed enums for `method` are `most_frequent`, `constant`"
        if method == "constant":
            assert val_const is not None, "Fill value must be provided for `constant`"
        self.__val_const = val_const
        self.__val_learned = {}
        self.__method = method
        self.__cols = []
        self.__pd_df = pd.DataFrame
    
    @property
    def val_const(self):
        return self.__val_const
    
    @property
    def val_learned(self):
        return self.__val_learned

    @property
    def method(self):
        return self.__method
    
    def get_feature_names(self):
        return self.__cols
    
    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        self.__cols = X_.columns
        if self.__method == "most_frequent":
            for column in X_.columns:
                self.__val_learned[column] = X_.loc[:, column].value_counts(ascending=False).index[0]
        elif self.__method == "constant":
            for column in X_.columns:
                self.__val_learned[column] = self.__val_const
        return self
    
    def transform(self, X):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        for column in X_.columns:
            X_.loc[X_[column].isnull(), column] = self.__val_learned[column]
        return X_