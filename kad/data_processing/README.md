# KAD Data Processing Module

## Overview

The Data Processing module provides a collection of utilities for transforming, validating, and processing time-series data, primarily focused on metrics retrieved from Prometheus. This module is part of the larger system and works together with the data sources and model components.

## Components

### Interface

#### `i_data_processor.py`
Defines the abstract interface `IDataProcessor` that all data processors must implement. This interface enforces a common contract through:
- `transform_data(input_df)`: Abstract method that takes a pandas DataFrame as input and returns a transformed DataFrame.

### Data Transformation Components

#### `composite_data_processor.py`
Implements the Composite pattern to allow chaining multiple data processors together:
- `CompositeDataProcessor`: Takes a list of `IDataProcessor` implementations and applies them sequentially
- Enables building complex data transformation pipelines from simple processing components

#### `downsampler.py`
Implements downsampling (reducing the frequency) of time-series data:
- `Downsampler`: Resamples data to a lower frequency (e.g., from seconds to minutes)
- Parameters:
  - `period`: Time period for resampling (e.g., '1min', '1h')
  - `agg_method`: Aggregation method to use (e.g., mean, sum, max)

#### `upsampler.py`
Implements upsampling (increasing the frequency) of time-series data:
- `Upsampler`: Resamples data to a higher frequency (e.g., from minutes to seconds)
- Uses backward fill (`bfill`) to fill new data points
- Parameters:
  - `period`: Target time period for resampling (e.g., '10s', '1s')

#### `ewm.py`
Implements exponentially weighted moving average smoothing:
- `Ewm`: Applies exponential weighting to smooth time-series data
- Parameters:
  - `com`: Center of mass parameter controlling the weighting (default: 0.5)

### Prometheus Integration

#### `prom_wrapper.py`
Provides a wrapper around the Prometheus API client:
- `PrometheusConnectWrapper`: Simplifies querying Prometheus for time-series data
- Methods:
  - `perform_query`: Executes PromQL queries over a specified time range
  - `fetch_metric_range_data`: Retrieves metrics with specific label configurations

#### `response_validator.py`
Validates responses from Prometheus queries:
- `validate`: Ensures a single metric was found in the response
- `MetricValidatorException`: Custom exception for metric validation errors

#### `metric_parser.py`
Utilities for converting Prometheus metrics to pandas DataFrames:
- `metric_to_dataframe`: Converts Prometheus metric response to a pandas DataFrame
- `split_dataset`: Splits a dataset into training and testing portions (70%/30% split)

## Usage

These components can be used individually or combined to create data processing pipelines for time-series data:

```python
# Example: Creating a processing pipeline
from kad.data_processing.composite_data_processor import CompositeDataProcessor
from kad.data_processing.downsampler import Downsampler
from kad.data_processing.ewm import Ewm

# Create individual processors
downsampler = Downsampler(period='1min', agg_method='mean')
smoother = Ewm(com=0.3)

# Create a composite processor that applies both transformations
pipeline = CompositeDataProcessor([downsampler, smoother])

# Apply the pipeline to data
processed_df = pipeline.transform_data(input_df)
```

## Integration with KAD System

This module is designed to work with:
- **Data Sources**: Processing data retrieved from Prometheus or other sources
- **Models**: Preparing data for anomaly detection models
- **Visualization**: Transforming data for effective visualization and analysis

The components follow a consistent interface, making them composable and extensible for various data processing needs in the context of Kubernetes monitoring and anomaly detection.
