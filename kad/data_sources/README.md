# KAD Data Sources

This package provides a framework for retrieving and preprocessing time-series data from different sources for analytics and modeling purposes.

## Overview

The package consists of three key components:

1. **IDataSource (i_data_source.py)**: An abstract interface that defines the common contract for all data sources.
2. **PrometheusDataSource (prom_data_source.py)**: Implementation for retrieving metrics from Prometheus monitoring systems.
3. **ExemplaryDataSource (exemplary_data_source.py)**: Implementation for loading and processing data from CSV files.

## Components

### IDataSource Interface

The base abstract class that all data sources must implement. It defines:

- `get_train_data()`: Retrieves the initial dataset for training purposes.
- `get_next_batch()`: Fetches new data incrementally.
- `update_last_processed_timestamp()`: Keeps track of data retrieval progress.

The interface also includes a custom `DataSourceException` for handling data source-specific errors.

### PrometheusDataSource

This implementation retrieves time-series metrics from a Prometheus monitoring system.

**Key features**:
- Connect to Prometheus endpoints
- Execute PromQL queries to retrieve metrics
- Process and validate the returned data
- Track time intervals to allow for incremental data retrieval

**Usage parameters**:
- `query`: PromQL query string
- `prom_url`: URL of the Prometheus server
- `metric_name`: Name of the metric to extract
- `start_time`: Beginning of the time range
- `stop_time`: End of the time range
- `update_interval_sec`: Time interval (in seconds) for incremental updates

### ExemplaryDataSource

This implementation loads time-series data from CSV files, providing a reference implementation that can also be used for testing or demonstration purposes.

**Key features**:
- Load data from CSV files
- Resample data to hourly intervals
- Process and validate the data
- Support incremental data retrieval to simulate real-time data sources

**Usage parameters**:
- `path`: Path to the CSV file
- `metric_name`: Name of the metric to extract (currently only supports "value")
- `start_time`: Beginning of the time range
- `stop_time`: End of the time range
- `update_interval_hours`: Time interval (in hours) for incremental updates

## Configuration

The system is configured using a YAML configuration file (`config.yaml`) that specifies:

### Prometheus Configuration
- `PROMETHEUS_URL`: URL of the Prometheus server (e.g., "http://localhost:9090")
- `START_TIME`: Beginning timestamp for data collection (e.g., "2021-11-08 18:00:00")
- `END_TIME`: Ending timestamp for data collection (e.g., "2021-11-08 19:00:00")
- `METRIC_NAME`: Complete PromQL query string for the metric to collect
- `UPDATE_INTERVAL_SEC`: Time interval in seconds for incremental data updates

### API Configuration
- `APP_URL`: Base URL for the application's API (e.g., "http://localhost:5000/")
- `UPDATE_DATA_ENDPOINT`: Endpoint for updating data
- `GET_RESULTS_ENDPOINT`: Endpoint for retrieving analysis results
- `PLOT_RESULTS_ENDPOINT`: Endpoint for generating plots
- `UPDATE_CONFIG_ENDPOINT`: Endpoint for updating configuration

### Model Configuration
- `MODEL_NAME`: The time series model to use (e.g., "SarimaModel")

This configuration file connects the data sources to an API interface and specifies the model used for analysis, creating a complete time-series analysis pipeline.

## Common Functionality

Both implementations share these key features:
- Time-based data retrieval and windowing
- Automatic calculation of time intervals between data points
- Support for both bulk (training) and incremental (streaming) data retrieval
- Error handling for missing or malformed data

## Dependencies

- pandas: For data manipulation and time-series handling
- logging: For operation logging
- datetime: For time manipulation
- numpy: For numerical operations (used in ExemplaryDataSource)

Additionally, the PrometheusDataSource depends on these modules (not included in the provided files):
- `kad.data_processing.prom_wrapper`: For connecting to Prometheus
- `kad.data_processing.response_validator`: For validating Prometheus responses
- `kad.data_processing.metric_parser`: For converting Prometheus metrics to pandas DataFrames

## Error Handling

The package provides custom exceptions:
- `DataSourceException`: For general data source errors
- `MetricValidatorException`: For metric validation errors (referenced in ExemplaryDataSource)
