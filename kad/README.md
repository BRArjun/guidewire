# Data Fetcher

## Overview
`Data Fetcher` is a script responsible for periodically requesting new data from an external API and running the core anomaly detection system. It ensures that the latest data is fetched, processed, and used for anomaly detection.

## Features
- **Periodic Data Fetching**: Uses APScheduler to request new data at specified intervals.
- **Configuration Management**: Loads settings from a YAML configuration file.
- **Error Handling**: Logs various errors related to API requests, malformed metrics, and data source issues.
- **Automatic Retry Mechanism**: If a failure occurs, the script waits for a predefined interval before retrying.

## Dependencies
Ensure the following libraries are installed before running the script:


### Configuration
The script reads configurations from `kad/config/config.yaml`. Ensure this file contains the required API URL and update interval settings

## Key Components
- **Data Requesting**: Sends a GET request to the API endpoint to fetch new data.
- **Scheduler**: Uses `BackgroundScheduler` to automate periodic data fetching.
- **Core Execution**: Initializes the `Core` component for processing the fetched data.
- **Logging**: Logs information, warnings, and errors to `/tmp/kad.log`.
- **Retry Mechanism**: If an error occurs, the script retries execution after `RETRY_INTERV` seconds.

## Error Handling
The script handles the following exceptions:
- **ConnectionError**: Logs failure when unable to reach the API.
- **MetricValidatorException**: Captures issues with malformed metric responses.
- **DataSourceException**: Handles errors due to insufficient training data.
- **General Exceptions**: Catches and logs any other unexpected errors.

