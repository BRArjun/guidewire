# Models Evaluator

## Overview
`ModelsEvaluator` is a Python class designed for evaluating anomaly detection models. It provides various evaluation metrics, visualization tools, and custom scoring mechanisms to assess model performance effectively.

## Features
- **Standard Metrics**: Computes accuracy, precision, recall, AUROC, and average precision.
- **Visualization Tools**:
  - Confusion Matrix
  - Precision-Recall Curve
  - ROC Curve
- **Custom Scoring Functions**:
  - Accuracy Scoring Component
  - Collective Scoring Component
  - Precision Scoring Component
- **Supports Time-Series Data Evaluation**

## Dependencies
Ensure the following libraries are installed before using the class:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Custom Scoring Functions
The `get_customized_score` method evaluates models based on three components:
1. **Accuracy Scoring Component** - Measures how well anomalies are detected in a time-series window.
2. **Collective Scoring Component** - Considers the effectiveness of detecting anomalies over time.
3. **Precision Scoring Component** - Evaluates the model's precision in anomaly detection.

