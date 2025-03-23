# Minikube Setup for Sock Shop & Monitoring Stack

## Overview
This script automates the setup of a Minikube cluster and deploys the Sock Shop microservices demo along with a monitoring stack. It ensures proper resource allocation, namespace creation, and service availability.

## Features
- **Deletes existing Minikube cluster** to ensure a fresh start.
- **Starts a new Minikube cluster** with specified CPU, memory, and disk size.
- **Creates namespaces** for Sock Shop (`sock-shop`) and monitoring (`monitoring`).
- **Deploys Sock Shop microservices** using a pre-defined Kubernetes manifest.
- **Deploys a monitoring stack** for observability.
- **Waits for all pods to be ready** before proceeding.
- **Sets up port forwarding** for Prometheus, Grafana, and the front-end service.
- **Annotates services** for Prometheus scraping.
- **Launches Minikube dashboard** for monitoring.

## Prerequisites
Ensure the following dependencies are installed before running the script:

```bash
minikube
kubectl
```

## Usage
To run the setup script, execute:

```bash
chmod +x setup.sh
./setup.sh
```

## Port Forwarding
Once the setup is complete, the following services will be accessible:
- **Prometheus**: `http://localhost:9090`
- **Sock Shop Front-End**: `http://localhost:30001`
- **Grafana**: `http://localhost:31300`

## Notes
- If the `microservices-demo` directory does not exist, the script will exit with an error.
- The script waits for pods to be ready before proceeding, with a timeout of 300 seconds.
- Port forwarding runs in the background, ensuring services remain accessible.
