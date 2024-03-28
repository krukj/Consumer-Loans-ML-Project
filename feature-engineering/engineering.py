from sklearn.base import BaseEstimator, TransformerMixin
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