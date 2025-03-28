## Running the application

### Locally

First, adjust the configuration file (`kad/config.yaml`), specifying the data source fot the system and metric that will be monitored.

Make sure that your Python version is `Python 3.6`. Then, install the requirements:

`pip install -r kad/requirements.txt`

Make sure that the repo is visible in PYTHONPATH:

`export PYTHONPATH=[PATH_TO_REPO_ROOT]`

Then you can start the application:

`python kad/main.py`

### K8s cluster

The configuration can be found in `kad/kad-configmap.yaml`.

To start the application in k8s cluster, run the following:

    kubectl apply -f kad-configmap.yaml
    kubectl apply -f kad-deployment.yaml
    kubectl apply -f kad-service.yaml

## Architecture

It consists of six independent modules: Core, Data Source Module, Data Processing Module, Model, Model Selector and Visualization Module. The core module connects all the other modules and manages the application. The Data Source Module provides a general interface that is a communication channel between KAD and the observed cluster. Currently, the system supports Prometheus data format. The Data Processing Module manipulates the data format and adjusts it to the requirements of Machine Learning Models. The Model Selector is a module that chooses a model that fits training data best. The Visualization Module gives some additional features related to the system output.

There are 4 available anomaly detection models that the user may choose from:

- Sarima
- HMM
- LSTM
- Autoencoder

The model can be either chosen manually or selected in an automatic procedure.

An HTTP API that allows to fetch data from the application and modify its configuration. It exposes configurable endpoints for both results fetching and configuration updates.

![kad](https://user-images.githubusercontent.com/39968023/139718190-0b40b106-ee77-4c8f-a412-ba558a7a0d6f.gif)

