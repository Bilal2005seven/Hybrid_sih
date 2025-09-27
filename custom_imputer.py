from sklearn.impute import KNNImputer
import numpy as np

class KNNImputerWithStore(KNNImputer):
    """Custom KNNImputer wrapper that stores fitted data for reuse."""
    
    def __init__(self, missing_values=np.nan, n_neighbors=5, weights="uniform", metric="nan_euclidean", copy=True, add_indicator=False):
        super().__init__(
            missing_values=missing_values,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            copy=copy,
            add_indicator=add_indicator
        )
        self.fitted_data_ = None
        self.last_imputed_ = None

    def fit(self, X, y=None):
        self.fitted_data_ = X
        return super().fit(X, y)
    
    def transform(self, X):
        # Store the imputed values for later access
        imputed_X = super().transform(X)
        self.last_imputed_ = imputed_X
        return imputed_X
    
    def fit_transform(self, X, y=None):
        # Store the imputed values for later access
        imputed_X = super().fit_transform(X, y)
        self.last_imputed_ = imputed_X
        return imputed_X