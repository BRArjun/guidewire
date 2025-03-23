# import datetime
# import io
# import json
# import logging
# import os
# import sys

# from kad.data_processing import composite_data_processor, upsampler
# from kad.model.model_utils import name_2_model
# from kad.model_selector import model_selector

# sys.path.insert(1, "..")
# from matplotlib import pyplot as plt
# import pandas as pd
# from flask import Flask, send_file, Response, jsonify, request
# from flask_cors import cross_origin, CORS

# from kad.data_sources import i_data_source
# from kad.data_sources.exemplary_data_source import ExemplaryDataSource
# from kad.data_sources.prom_data_source import PrometheusDataSource
# from kad.data_sources.i_data_source import DataSourceException
# from kad.kad_utils import kad_utils
# from kad.kad_utils.kad_utils import EndpointAction
# from kad.model import i_model
# from kad.visualization.visualization import visualize
# from prometheus_flask_exporter import PrometheusMetrics
# import time
# from prometheus_client import Summary


# class Core(object):
#     app = None

#     def __init__(self, p_config: dict):
#         self.app = Flask("KAD app")
#         self.metrics = PrometheusMetrics(self.app)

#         CORS(self.app, resources={r"/*": {"origins": "*"}})

#         self.config: dict = p_config

#         self.app.add_url_rule(rule="/" + self.config["PLOT_RESULTS_ENDPOINT"],
#                               endpoint=self.config["PLOT_RESULTS_ENDPOINT"],
#                               view_func=self.plot_results)
#         self.app.add_url_rule(rule="/" + self.config["GET_RESULTS_ENDPOINT"],
#                               endpoint=self.config["GET_RESULTS_ENDPOINT"],
#                               view_func=self.get_results)
#         self.app.add_url_rule(rule="/" + self.config["UPDATE_DATA_ENDPOINT"],
#                               endpoint=self.config["UPDATE_DATA_ENDPOINT"],
#                               view_func=self.update_data)
#         self.app.add_url_rule(rule="/" + self.config["UPDATE_CONFIG_ENDPOINT"],
#                               endpoint=self.config["UPDATE_CONFIG_ENDPOINT"],
#                               view_func=self.update_config,
#                               methods=["POST"])

#         self.data_source: i_data_source = None
#         self.model_selector: model_selector.ModelSelector = None
#         self.model: i_model.IModel = None
#         self.last_train_sample = None
#         self.results_df: pd.DataFrame = None
#         self.train_mean = None
#         self.train_std = None

#         self.set_up()

#     def reset(self):
#         self.data_source = None
#         self.model = None
#         self.last_train_sample = None
#         self.results_df = None
#         self.train_mean = None
#         self.train_std = None

#     def set_up(self):
#         # Exemplary data
#         # file = "data/archive/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv"
#         # daily_jumpsup_csv_path = os.path.join(
#         #     "/home/maciek/Documents/Magisterka/kubernetes-anomaly-detector/notebooks/",
#         #     file)
#         # self.data_source = ExemplaryDataSource(
#         #     path=daily_jumpsup_csv_path,
#         #     metric_name=self.config["METRIC_NAME"],
#         #     start_time=datetime.datetime.strptime(self.config["START_TIME"], "%Y-%m-%d %H:%M:%S"),
#         #     stop_time=datetime.datetime.strptime(self.config["END_TIME"], "%Y-%m-%d %H:%M:%S"),
#         #     update_interval_hours=10)
#         self.data_source = PrometheusDataSource(query=self.config["METRIC_NAME"],
#                                                 metric_name=self.config["METRIC_NAME"],
#                                                 prom_url=self.config["PROMETHEUS_URL"],
#                                                 start_time=datetime.datetime.strptime(self.config["START_TIME"],
#                                                                                       "%Y-%m-%d %H:%M:%S"),
#                                                 stop_time=datetime.datetime.strptime(self.config["END_TIME"],
#                                                                                      "%Y-%m-%d %H:%M:%S"),
#                                                 update_interval_sec=self.config["UPDATE_INTERVAL_SEC"])

#         # self.model = SarimaModel(order=(0, 0, 0), seasonal_order=(1, 0, 1, 24))
#         # self.model = AutoEncoderModel(time_steps=12)
#         # self.model = HmmModel()
#         self.results_df = None
#         self.last_train_sample = None

#     def __get_train_data(self) -> pd.DataFrame:
#         train_df = self.data_source.get_train_data()

#         train_df.to_pickle("train_df.pkl")

#         # data_proc = composite_data_processor.CompositeDataProcessor(
#         #     [upsampler.Upsampler("10s")])
#         # train_df = data_proc.transform_data(train_df)

#         self.last_train_sample = len(train_df)
#         self.train_mean = train_df.mean()
#         self.train_std = train_df.std()
#         return kad_utils.normalize(train_df, self.train_mean, self.train_std)

#     def __select_and_train_model(self):
#         train_df = self.__get_train_data()

#         self.model_selector = model_selector.ModelSelector(train_df)
#         if "MODEL_NAME" in self.config and self.config["MODEL_NAME"] != "":
#             self.model = name_2_model(self.config["MODEL_NAME"], self.model_selector)
#         else:
#             self.model = self.model_selector.select_model()
#         logging.info("Selected model: " + self.model.__class__.__name__)

#         self.model.train(train_df)  # TODO remove or add a separate option w/o extra validation

#         if len(train_df) < 2:
#             logging.warning("Almost empty training df (len < 2)")

#         logging.info("Model trained")

#     def test(self, test_df):
#         self.results_df = self.model.test(test_df)

#     def get_latest_image(self):
#         if self.results_df is None:
#             logging.warning("Results not obtained yet")
#             return None

#         visualize(self.results_df, self.config["METRIC_NAME"], self.last_train_sample)

#         bytes_image = io.BytesIO()
#         plt.savefig(bytes_image, format="png")
#         bytes_image.seek(0)
#         return bytes_image

#     def run(self, p_scheduler):
#         self.__select_and_train_model()
#         p_scheduler.start()
#         self.app.run(debug=True, threaded=False)

#     def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None):
#         self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler))

#     @cross_origin(supports_credentials=True)
#     def plot_results(self):
#         logging.info("Results plot requested")
#         bytes_obj = self.get_latest_image()

#         if bytes_obj is None:
#             logging.warning("Image empty!")
#             return Response(status=404, headers={})

#         return send_file(bytes_obj,
#                          attachment_filename="plot.png",
#                          mimetype="image/png")

#     @cross_origin(supports_credentials=True)
#     def get_results(self):
#         logging.info("Results in raw format requested")

#         if self.results_df is None:
#             logging.warning("Results not obtained yet")
#             return Response(status=404, headers={})

#         results_json = json.loads(self.results_df.to_json())
#         results_json["metric"] = self.config["METRIC_NAME"]
#         results_json["model"] = self.model.__class__.__name__
#         return jsonify(results_json)

#     @cross_origin(supports_credentials=True)
#     def update_data(self):
#         logging.info("Updating data")

#         if not self.model or not self.model.is_trained():
#             logging.warning("Model not trained while requesting new data")
#             return Response(status=200, headers={})

#         if self.data_source is None:
#             logging.warning("Data source not present while requesting data update")
#             return Response(status=200, headers={})

#         try:
#             new_data = self.data_source.get_next_batch()
#         except DataSourceException as dsexc:
#             logging.warning(str(dsexc))
#             return Response(status=200, headers={})

#         if len(new_data) == 0:
#             logging.warning("No new data has been obtained")
#             return Response(status=200, headers={})

#         new_data = kad_utils.normalize(new_data, self.train_mean, self.train_std)

#         try:
#             self.test(new_data)
#             self.data_source.update_last_processed_timestamp()
#         except Exception as exc:
#             logging.warning("Impossible to test: " + str(exc))

#         return Response(status=200, headers={})

#     @cross_origin(supports_credentials=True)
#     def update_config(self):
#         logging.info("Updating config")
#         json_data = request.get_json()
#         print(json_data)

#         if not self.are_changes_in_config(json_data):
#             logging.warning("No changes in config")
#             return Response(status=200, headers={})

#         logging.info("Changes in config - update needed")
#         try:
#             self.config["METRIC_NAME"] = json_data["METRIC_NAME"]
#             self.config["START_TIME"] = json_data["START_TIME"]
#             self.config["END_TIME"] = json_data["END_TIME"]
#             if "MODEL_NAME" in json_data.keys():
#                 self.config["MODEL_NAME"] = json_data["MODEL_NAME"]
#             else:
#                 self.config["MODEL_NAME"] = ""
#             self.set_up()

#             logging.info("Training once again")
#             self.__select_and_train_model()
#         except Exception as ex:
#             logging.error("Unsuccessful config update - resetting config. Exception: " + str(ex))
#             self.reset()
#             return jsonify({"Error": "Unsuccessfull config update - resetting config"})

#         logging.info("Update config successful")
#         return jsonify({"train_df_len": self.last_train_sample})

#     def are_changes_in_config(self, new_config: dict) -> bool:
#         shared_items = {k: self.config[k] for k in self.config if k in new_config and self.config[k] == new_config[k]}

#         return len(shared_items) != len(new_config.keys())


import datetime
import io
import json
import logging
import os
import sys
import argparse
import numpy as np
import pandas as pd
from flask import Flask, send_file, Response, jsonify, request
from flask_cors import CORS
from matplotlib import pyplot as plt
from prometheus_flask_exporter import PrometheusMetrics

# Mock data processing and model components
class CompositeDataProcessor:
    def transform_data(self, data):
        return data

class Upsampler:
    def __init__(self, freq):
        pass

class DataSourceException(Exception):
    pass

class IDataSource:
    def get_train_data(self):
        raise NotImplementedError
        
    def get_next_batch(self):
        raise NotImplementedError

class KadUtils:
    @staticmethod
    def normalize(data, mean, std):
        return (data - mean) / std
        
    @staticmethod
    def calculate_validation_err(predictions, ground_truth):
        return np.mean(np.abs(predictions - ground_truth))

kad_utils = KadUtils()

# Model implementations
class IModel:
    def __init__(self):
        self.trained = False
        
    def train(self, data):
        pass
        
    def test(self, data):
        pass

class AutoEncoderModel(IModel):
    def __init__(self, time_steps=12, batch_size=12, learning_rate=0.001):
        super().__init__()
        self.time_steps = time_steps
        
    def train(self, data):
        self.trained = True
        return 0.1
        
    def test(self, data):
        results = data.copy()
        results['predictions'] = data['value'] * 0.9
        results['error'] = np.abs(results['value'] - results['predictions'])
        results['anomalies'] = results['error'] > 0.5
        return results

class LstmModel(IModel):
    def __init__(self, time_steps=12, batch_size=12):
        super().__init__()
        
    def train(self, data):
        self.trained = True
        return 0.15
        
    def test(self, data):
        results = data.copy()
        results['predictions'] = data['value'] * 0.85
        results['error'] = np.abs(results['value'] - results['predictions'])
        results['anomalies'] = results['error'] > 0.6
        return results

class SarimaModel(IModel):
    def __init__(self, order=(1,0,0), seasonal_order=(1,0,1,12)):
        super().__init__()
        
    def train(self, data):
        self.trained = True
        return 0.2
        
    def test(self, data):
        results = data.copy()
        results['predictions'] = data['value'].rolling(5).mean()
        results['error'] = np.abs(results['value'] - results['predictions'])
        results['anomalies'] = results['error'] > 0.4
        return results

class HmmModel(IModel):
    def train(self, data):
        self.trained = True
        return 0.25
        
    def test(self, data):
        results = data.copy()
        results['predictions'] = data['value'] * 0.95
        results['error'] = np.abs(results['value'] - results['predictions'])
        results['anomalies'] = results['error'] > 0.3
        return results

MODELS = {
    'autoencoder': AutoEncoderModel,
    'lstm': LstmModel,
    'sarima': SarimaModel,
    'hmm': HmmModel
}

class DummyDataSource(IDataSource):
    def __init__(self):
        self.last_timestamp = datetime.datetime(2023, 1, 1)
        self.data = self._generate_data(1000)
        
    def _generate_data(self, n_points):
        time = np.arange(n_points)
        base = np.sin(0.1 * time)
        noise = 0.1 * np.random.randn(n_points)
        anomalies = 0.5 * (np.random.rand(n_points) > 0.95)
        return pd.DataFrame({
            'timestamp': [self.last_timestamp + datetime.timedelta(minutes=5*i) for i in range(n_points)],
            'value': base + noise + anomalies
        }).set_index('timestamp')
        
    def get_train_data(self):
        return self.data.iloc[:700]
        
    def get_next_batch(self):
        new_data = self._generate_data(10)
        self.data = pd.concat([self.data, new_data])
        return new_data

class Core:
    def __init__(self, config):
        self.app = Flask("KAD")
        self.metrics = PrometheusMetrics(self.app)
        CORS(self.app)
        
        self.config = config
        self.data_source = DummyDataSource()
        self.model = MODELS[self.config['MODEL_NAME'].lower()]()
        self.results_df = None
        
        self.app.add_url_rule('/plot', 'plot', self.plot_results)
        self.app.add_url_rule('/results', 'results', self.get_results)
        self.app.add_url_rule('/update', 'update', self.update_data)
        self.app.add_url_rule('/update_config', 'update_config', 
                            self.update_config, methods=['POST'])
                            
    def train_model(self):
        train_df = self.data_source.get_train_data()
        self.model.train(train_df)
        
    def plot_results(self):
        if self.results_df is None:
            return Response(status=404)
            
        plt.figure()
        plt.plot(self.results_df['value'], label='Actual')
        plt.plot(self.results_df['predictions'], label='Predicted')
        plt.scatter(self.results_df[self.results_df['anomalies']].index,
                    self.results_df[self.results_df['anomalies']]['value'],
                    color='red', label='Anomalies')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
        
    def get_results(self):
        if self.results_df is None:
            return Response(status=404)
        return jsonify(self.results_df.tail(100).to_dict())
        
    def update_data(self):
        new_data = self.data_source.get_next_batch()
        self.results_df = self.model.test(new_data)
        return Response(status=200)
        
    def update_config(self):
        data = request.get_json()
        new_model = data.get('model', '').lower()
        
        if new_model in MODELS:
            self.config['MODEL_NAME'] = new_model
            self.model = MODELS[new_model]()
            self.train_model()
            return jsonify({'status': 'Model updated'})
            
        return jsonify({'error': 'Invalid model'}), 400
        
    def run(self):
        self.train_model()
        self.app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=MODELS.keys(),
                       default='autoencoder', help='Model to use for detection')
    args = parser.parse_args()

    config = {
        'MODEL_NAME': args.model,
        'UPDATE_INTERVAL_SEC': 300
    }

    logging.basicConfig(level=logging.INFO)
    core = Core(config)
    core.run()