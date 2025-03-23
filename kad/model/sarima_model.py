# sarima_model.py
import logging
import warnings
from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
import kad.kad_utils.kad_utils as kad_utils
from kad.model import i_model

class SarimaModel(i_model.IModel):
    def __init__(self, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int], train_valid_ratio=0.7):
        super().__init__()
        self.model = None
        self.model_results = None
        self.error_threshold: float = 0.0
        self.anomaly_score_threshold: float = 0.95
        self.results_df = None
        self.order = order
        self.seasonal_order = seasonal_order
        self.train_valid_ratio = train_valid_ratio

    @staticmethod
    def __calculate_threshold(valid_errors: np.ndarray) -> float:
        return np.max(valid_errors)

    def train(self, train_df: pd.DataFrame) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            tr_df, valid_df = train_test_split(train_df, shuffle=False, train_size=self.train_valid_ratio)
            self.model = SARIMAX(tr_df.values,
                                 exog=None,
                                 order=self.order,
                                 seasonal_order=self.seasonal_order,
                                 enforce_stationarity=True,
                                 enforce_invertibility=False)
            self.model_results = self.model.fit()

            forecast: np.ndarray = self.model_results.forecast(len(valid_df))
            ground_truth = valid_df.to_numpy().flatten()
            
            # Handle model results update
            try:
                self.model_results = self.model_results.append(ground_truth)
            except AttributeError:
                self.model_results = self.model_results.extend(ground_truth)

            abs_error = np.abs(forecast - ground_truth)
            self.error_threshold = self.__calculate_threshold(abs_error)

            # Initialize results dataframe with proper columns
            self.results_df = train_df.copy()
            self.results_df[kad_utils.PREDICTIONS_COLUMN] = np.nan
            self.results_df[kad_utils.ERROR_COLUMN] = np.nan
            self.results_df[kad_utils.ANOM_SCORE_COLUMN] = np.nan
            self.results_df[kad_utils.ANOMALIES_COLUMN] = False

            # Assign values to the last part of the dataframe
            last_idx = len(valid_df)
            self.results_df.iloc[-last_idx:, self.results_df.columns.get_loc(kad_utils.PREDICTIONS_COLUMN)] = forecast
            self.results_df.iloc[-last_idx:, self.results_df.columns.get_loc(kad_utils.ERROR_COLUMN)] = abs_error

            logging.info(f"SARIMA anomaly threshold set to: {self.error_threshold:.4f}")
            self.trained = True
            return kad_utils.calculate_validation_err(forecast, ground_truth)

    def test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Generate forecast and errors
            forecast = self.model_results.forecast(len(test_df))
            abs_error = np.abs(forecast - test_df.to_numpy().flatten())

            # Prepare test_df with required columns
            test_df = test_df.copy()
            test_df[kad_utils.PREDICTIONS_COLUMN] = np.nan
            test_df[kad_utils.ERROR_COLUMN] = np.nan
            test_df[kad_utils.ANOM_SCORE_COLUMN] = np.nan
            test_df[kad_utils.ANOMALIES_COLUMN] = False

            # Update the test_df with new values
            test_df.loc[:, kad_utils.PREDICTIONS_COLUMN] = forecast
            test_df.loc[:, kad_utils.ERROR_COLUMN] = abs_error
            test_df.loc[:, kad_utils.ANOM_SCORE_COLUMN] = kad_utils.calculate_anomaly_score(
                abs_error, self.error_threshold
            )
            test_df.loc[:, kad_utils.ANOMALIES_COLUMN] = (
                test_df[kad_utils.ANOM_SCORE_COLUMN] >= self.anomaly_score_threshold
            )

            # Concatenate results
            self.results_df = pd.concat([self.results_df, test_df], axis=0)

            # Update model results
            if np.any(test_df[kad_utils.ANOMALIES_COLUMN]):
                try:
                    self.model_results = self.model_results.append(forecast)
                except AttributeError:
                    self.model_results = self.model_results.extend(forecast)
            else:
                try:
                    self.model_results = self.model_results.append(test_df.values)
                except AttributeError:
                    self.model_results = self.model_results.extend(test_df.values)

            return self.results_df

    def get_results(self) -> pd.DataFrame:
        return self.results_df.copy()
