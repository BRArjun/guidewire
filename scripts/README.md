# Configuration Update Script

## Overview
This script benchmarks the response time of a configuration update request to a server endpoint. It sends repeated POST requests with different time intervals and logs the request durations.

## Features
- **Sends configuration update requests** to a specified endpoint.
- **Measures request duration** for each request.
- **Repeats the process** for multiple probes and repetitions to collect performance metrics.
- **Logs results in a dictionary** mapping dataset length to request durations.
- **Saves the results** in a pickle file (`eval.pkl`) for further analysis.

## Dependencies
Ensure you have the following dependencies installed before running the script:

## Configuration
Modify the following parameters in the script as needed:
- **`NUM_PROBES`**: Number of probe iterations.
- **`NUM_REPETITIONS`**: Number of repetitions per probe.
- **`start_time` / `stop_time`**: Defines the time range for the update request.
- **`url`**: Set the API endpoint for the configuration update.

## Output
- The script prints the response JSON from the server.
- Stores the request duration data in `eval.pkl` using the `pickle` module.
- The request duration is mapped to dataset length for further evaluation.
