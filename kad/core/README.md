# Core.py

## Overview

`core.py` is the central component of the system. It implements a Flask-based web application that acts as the orchestration layer for retrieving time-series data, training anomaly detection models, and providing analysis results through a REST API.

## Functionality

### Main Features

- **Web API**: Provides HTTP endpoints for interacting with the anomaly detection system
- **Model Selection**: Automatically selects and trains the appropriate time-series model
- **Data Processing**: Coordinates data retrieval, normalization, and processing
- **Visualization**: Generates plots of anomaly detection results
- **Configuration Management**: Supports dynamic configuration updates

### Core Class

The file defines a `Core` class that encapsulates all the functionality:

- **Initialization**: Sets up Flask app, metrics, CORS, and endpoints
- **Data Management**: Connects to data sources (Prometheus or exemplary) and processes data
- **Model Management**: Selects, initializes, and trains anomaly detection models
- **Results Handling**: Processes and visualizes detection results
- **Configuration**: Handles dynamic configuration updates

## API Endpoints

The `Core` class exposes several REST endpoints (configured via config.yaml):

- **Plot Results**: Returns a visualization of the anomaly detection results as a PNG image
- **Get Results**: Returns the raw anomaly detection results in JSON format
- **Update Data**: Triggers the retrieval of new data and updates the model results
- **Update Config**: Allows dynamic updates to the configuration

## Implementation Details

### Data Flow

1. Data is retrieved from a data source (typically Prometheus)
2. The data is normalized using mean and standard deviation of the training set
3. A model is selected and trained on the normalized data
4. New data is periodically retrieved and tested against the model
5. Results are made available through the API endpoints

### Integration Points

The system integrates with:

- **Data Sources**: Via the `i_data_source.IDataSource` interface
- **Models**: Via the `i_model.IModel` interface and `model_selector.ModelSelector`
- **Prometheus**: For both data collection and monitoring of the application itself

### Error Handling

The system includes comprehensive error handling:

- Logging of warnings and errors
- Appropriate HTTP response codes
- Exception handling for data source and model errors
- Recovery mechanisms for configuration updates

## Dependencies

- **Flask**: Web framework for the REST API
- **Flask-CORS**: Cross-Origin Resource Sharing support
- **Pandas**: Data manipulation and processing
- **Matplotlib**: Visualization generation
- **prometheus_flask_exporter**: Prometheus metrics for the Flask application
- **KAD Components**: Various internal modules for data processing, model selection, and visualization

## Usage

The `Core` class is intended to be instantiated with a configuration dictionary and run with a scheduler:

```python
core = Core(config_dict)
core.run(scheduler)
```

The scheduler is responsible for periodically triggering the `update_data` endpoint to fetch new data and update results.
