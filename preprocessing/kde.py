import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


class KernelDensityEstimator:
    def __init__(
        self,
        dataset: pd.DataFrame,
        params: dict = {"bandwidth": np.logspace(-4, 0, 20)},
    ):
        self.dataset = dataset.values
        self.params = params
        self.grid = None
        self.kde = None

    def get_bandwidth(self, cv: int = 3):
        """Grid search to find the optimal bandwidth for the kernel density estimator"""
        grid = GridSearchCV(
            KernelDensity(),
            self.params,
            cv=cv,
            verbose=sys.maxsize,
            n_jobs=-1,
        )
        return grid

    def fit(self):
        """Fit the kernel density estimator to the dataset"""
        self.grid = self.get_bandwidth()
        self.grid.fit(self.dataset)
        self.kde = self.grid.best_estimator_
        return self.kde.fit(self.dataset)

    def sample(self, n_samples: int = 10**4):
        """Sample from solved kernel density estimator"""
        data_frame = pd.DataFrame(
            index=np.arange(n_samples),
            columns=["moneyness", "time_to_maturity"],
            dtype="float64",
        )

        count: int = 0
        while count < n_samples:
            remaining = n_samples - count
            raw_samples = self.kde.sample(remaining)
            filtered = (
                (raw_samples[:, 0] > 0.75)
                & (raw_samples[:, 0] < 1.2)
                & (raw_samples[:, 1] > 0)
                & (raw_samples[:, 1] < 0.25)
            )
            valid = raw_samples[filtered]
            nvalid = len(valid)
            data_frame.iloc[count : count + nvalid, :] = valid
            count += nvalid

        return data_frame
