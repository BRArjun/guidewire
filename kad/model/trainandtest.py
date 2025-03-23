# test_models.py
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

# Import your model classes
from autoencoder_model import AutoEncoderModel
from hmm_model import HmmModel
from lstm_model import LstmModel
from sarima_model import SarimaModel
from i_model import IModel, ModelException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyDataSource:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.data = self._generate_data()
        
    def _generate_data(self):
        """Generate synthetic time series data with anomalies"""
        np.random.seed(42)
        time = np.arange(self.n_samples)
        base = np.sin(0.1 * time)
        noise = 0.1 * np.random.randn(self.n_samples)
        
        # Add anomalies
        anomalies = np.zeros(self.n_samples)
        anomaly_indices = np.random.choice(self.n_samples, size=20, replace=False)
        anomalies[anomaly_indices] = 2.0  # Large spikes
        
        timestamps = [datetime(2023, 1, 1) + timedelta(minutes=5*i) for i in range(self.n_samples)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': base + noise + anomalies
        }).set_index('timestamp')

    def get_train_data(self):
        return self.data.iloc[:800]  # First 80% for training

    def get_test_data(self):
        return self.data.iloc[800:]   # Last 20% for testing

def test_model(model: IModel, train_data: pd.DataFrame, test_data: pd.DataFrame):
    """Train and test a model with visualization"""
    try:
        # Train the model
        logger.info(f"Training {model.__class__.__name__}...")
        train_error = model.train(train_data)
        logger.info(f"Training completed. Validation error: {train_error:.4f}")

        # Test the model
        logger.info("Testing...")
        results = model.test(test_data)
        logger.info("Testing completed.")

        # Visualize results
        plt.figure(figsize=(15, 6))
        plt.plot(results.index, results['value'], label='Actual')
        plt.plot(results.index, results['predictions'], label='Predicted', alpha=0.7)
        
        anomalies = results[results['anomalies']]
        plt.scatter(anomalies.index, anomalies['value'], 
                   color='red', label='Anomalies')
        
        plt.title(f"{model.__class__.__name__} Results")
        plt.legend()
        plt.show()

        # Print anomaly statistics
        num_anomalies = anomalies.shape[0]
        logger.info(f"Detected {num_anomalies} anomalies in test data")
        if num_anomalies > 0:
            logger.info("Sample anomalies:")
            logger.info(anomalies.head(5))

    except ModelException as e:
        logger.error(f"Model error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Test anomaly detection models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['autoencoder', 'lstm', 'sarima', 'hmm'],
                       help='Model to test (autoencoder, lstm, sarima, hmm)')
    args = parser.parse_args()

    # Generate dummy data
    data_source = DummyDataSource()
    train_data = data_source.get_train_data()
    test_data = data_source.get_test_data()

    # Initialize selected model
    if args.model == 'autoencoder':
        model = AutoEncoderModel(time_steps=24, learning_rate=0.001)
    elif args.model == 'lstm':
        model = LstmModel(time_steps=24, batch_size=12)
    elif args.model == 'sarima':
        model = SarimaModel(order=(1, 0, 0), seasonal_order=(1, 0, 1, 24))
    elif args.model == 'hmm':
        model = HmmModel()
    else:
        raise ValueError("Invalid model selection")

    # Run test
    test_model(model, train_data, test_data)

if __name__ == "__main__":
    main()
