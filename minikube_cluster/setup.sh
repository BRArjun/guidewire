#!/bin/bash

set -e  # Exit on error

echo "Deleting existing Minikube cluster..."
minikube delete

echo "Starting Minikube with adjusted resources..."
minikube start --memory=4096 --cpus=2 --disk-size=20g

echo "Creating necessary namespaces..."
kubectl create namespace sock-shop || echo "Namespace sock-shop already exists"
kubectl create namespace monitoring || echo "Namespace monitoring already exists"

echo "Setting up kubeconfig..."
mkdir -p secret
cat ~/.kube/config > ./secret/kubeconfig

echo "Checking if the microservices-demo directory exists..."
if [ ! -d "../microservices-demo-master/deploy/kubernetes" ]; then
    echo "Error: The directory '../microservices-demo-master/deploy/kubernetes' does not exist."
    exit 1
fi

echo "Deploying Sock Shop microservices..."
kubectl apply -f ../microservices-demo-master/deploy/kubernetes/complete-demo.yaml -n sock-shop

echo "Deploying monitoring stack..."
kubectl apply -f ../microservices-demo-master/deploy/kubernetes/manifests-monitoring -n monitoring

echo "Waiting for all pods in sock-shop to be ready..."
timeout=300
interval=10
elapsed=0

while [[ $(kubectl get pods -n sock-shop --field-selector=status.phase!=Running 2>/dev/null | wc -l) -gt 0 && $elapsed -lt $timeout ]]; do
    echo "Waiting for sock-shop pods... ($elapsed/$timeout seconds elapsed)"
    sleep $interval
    elapsed=$((elapsed + interval))
done

if [[ $elapsed -ge $timeout ]]; then
    echo "Error: Some sock-shop pods failed to become ready within $timeout seconds."
    kubectl get pods -n sock-shop
    exit 1
fi

echo "Waiting for all pods in monitoring to be ready..."
elapsed=0
while [[ $(kubectl get pods -n monitoring --field-selector=status.phase!=Running 2>/dev/null | wc -l) -gt 0 && $elapsed -lt $timeout ]]; do
    echo "Waiting for monitoring pods... ($elapsed/$timeout seconds elapsed)"
    sleep $interval
    elapsed=$((elapsed + interval))
done

if [[ $elapsed -ge $timeout ]]; then
    echo "Error: Some monitoring pods failed to become ready within $timeout seconds."
    kubectl get pods -n monitoring
    exit 1
fi

echo "Setting up port forwarding..."
kubectl port-forward service/prometheus 9090 -n monitoring &
kubectl port-forward service/front-end 30001:80 -n sock-shop &
kubectl port-forward service/grafana 31300:80 -n monitoring &

echo "Annotating Sock Shop services for Prometheus scraping..."
kubectl annotate service -n sock-shop prometheus.io/scrape='true' --all || echo "Annotation failed, continuing..."

# Ensure Minikube's local Docker environment is used
eval $(minikube docker-env)

echo "------------------------"
echo "Launching Minikube dashboard..."
minikube dashboard &

echo "Setup completed successfully!"

