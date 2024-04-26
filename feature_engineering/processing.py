import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Simple column remover to remove low variance columns
class FeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(self.columns_to_drop,axis=1)

# Update name of the column
class FeatureNameUpdater(BaseEstimator, TransformerMixin):
    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X.rename(columns = {self.old_name: self.new_name}, inplace = True)
        
        return X
    
# Imputes missing values with mode and according to distribution of columns
class MyImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_mode_imputation, cols_dist_imputation):
        self.modes = {}
        self.distributions_of_columns = {}
        self.cols_mode_imputation = cols_mode_imputation
        self.cols_dist_imputation = cols_dist_imputation
    
    # Function to create dictionary containing distribution of a categorical column
    def get_col_distribution(self, X, col_name):
        value_counts = X[col_name].value_counts()
        number_of_missing_values = value_counts["Missing"]
        value_counts_dict = value_counts[1:].to_dict()
        
        # change to probabilities
        for key in value_counts_dict:
            value_counts_dict[key] = value_counts_dict[key] / (len(X) - number_of_missing_values)

        # if probabilites do not sum to 1 due to numerical errors
        if 0.99 < sum(value_counts_dict.values()) < 1:
            max_value = max(value_counts_dict.values())
            max_key = [key for key, value in value_counts_dict.items() if value == max_value][0]

            value_counts_dict[max_key] += 1 - sum(value_counts_dict.values())

        return value_counts_dict

    def fit(self, X, y=None):
        for col_name in self.cols_mode_imputation:
            self.modes[col_name] = X[col_name].mode()[0]
        
        for col_name in self.cols_dist_imputation:
            self.distributions_of_columns[col_name] = self.get_col_distribution(X, col_name)
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col_name in self.cols_mode_imputation:
            X_copy[col_name] = X[col_name].replace('Missing', self.modes[col_name])
        
        for col_name in self.cols_dist_imputation:
            column_distribution = self.distributions_of_columns[col_name]
            
            X_copy[col_name] = X[col_name].replace('Missing',
                                              np.random.choice(list(column_distribution.keys()), 
                                                        p = list(column_distribution.values())))
        
        return X_copy

# Encoder for categorical variables handling both ordered and unordered ones
class MyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_features_wo_order, cat_feature_with_order, categories_order):
        self.cat_features_wo_order = cat_features_wo_order
        self.cat_feature_with_order = cat_feature_with_order
        self.categories_order = categories_order
        self.ordinal_encoder = OrdinalEncoder(categories=[categories_order])

    def fit(self, X, y=None):
        self.ordinal_encoder.fit(X[[self.cat_feature_with_order]])

        return self

    def transform(self, X, y=None):
        X_copy = pd.get_dummies(X, columns=self.cat_features_wo_order)

        X_copy[self.cat_feature_with_order] = self.ordinal_encoder.transform(X[[self.cat_feature_with_order]])

        return X_copy

# scaling the data, based on choice it will either standarize(default) or normalize,
class MyScaler(BaseEstimator, TransformerMixin):
    def __init__(self, standarize=True):
        self.standarize = standarize
        
        if self.standarize is True:
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        # we want to scale only numerical columns
        X_num_cols = X.select_dtypes(include=['float64', 'int64'])
        
        self.scaler.fit(X_num_cols)
        
        return self

    def transform(self, X, y=None):
        X_num_cols = X.select_dtypes(include=['float64', 'int64'])
        
        # transform numerical columns
        X_num_cols_transformed = self.scaler.transform(X_num_cols)
        
        # change to df to access columns
        X_num_cols_transformed_df = pd.DataFrame(X_num_cols_transformed, columns=X_num_cols.columns, index=X_num_cols.index)

        # change them in X
        X_copy = X.copy()
        for col_name in X_num_cols_transformed_df.columns:
            X_copy[col_name] = X_num_cols_transformed_df[col_name]

        return X_copy
    

class OutlierReplacer(BaseEstimator, TransformerMixin):
    # It will replace outliers from columns based on provided dictionary(for now list of columns)
    # If to_remove_dict[col_name] is true outliers will be removed (and exchanged with a proper quantile)
    # It is possible to adjust parameter k, by default its set to 1.5 as usually it is done
    def __init__(self, columns=None, k=1.5):
        self.columns = columns
        self.k = k
        self.lower_bounds = {}
        self.upper_bounds = {}
        
    def fit(self, X, y=None):
        # Compute lower and upper bounds for each specified column
        if self.columns is None:
            self.columns = X.columns
        for col in self.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.k * iqr
            upper_bound = q3 + self.k * iqr
            self.lower_bounds[col] = lower_bound
            self.upper_bounds[col] = upper_bound
        return self
    
    def transform(self, X, y=None):
        # Replace outliers in specified columns with calculated bounds
        X_copy = X.copy()
        for col in self.columns:
            lower_bound = self.lower_bounds[col]
            upper_bound = self.upper_bounds[col]
            X_copy[col] = X_copy[col].clip(lower_bound, upper_bound)
        return X_copy

class CreateRatioFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X["WORK_SENIORITY_to_BUSINESS_AGE"] = X["WORK_SENIORITY"] / X["BUSINESS_AGE"]
        X["LENGTH_RELATIONSHIP_WITH_CLIENT_to_AGE"] = X["LENGTH_RELATIONSHIP_WITH_CLIENT"] / X["AGE"]

        max_value_A = X["WORK_SENIORITY_to_BUSINESS_AGE"].max()
        max_value_B = X["LENGTH_RELATIONSHIP_WITH_CLIENT_to_AGE"].max()

        X["WORK_SENIORITY_to_BUSINESS_AGE"].fillna(max_value_A, inplace=True)
        X["LENGTH_RELATIONSHIP_WITH_CLIENT_to_AGE"].fillna(max_value_B, inplace=True)

        return X
    
    def set_output(self, *args, **kwargs):
        return self