# Centralized and Federated Learning Examples on Human Activity Recognition

## Overview

This repository contains Jupyter notebooks that provide examples of both centralized and federated learning using the [Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

The aim is to demonstrate the differences and implementation details between centralized learning, where data is combined and processed on a single server, and federated learning, where data remains on decentralized devices and only model updates are shared.

## Dataset

The dataset used in these examples is the [Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) from the UCI Machine Learning Repository. This dataset contains sensor signals collected from smartphones of 30 subjects performing six different activities (walking, walking upstairs, walking downstairs, sitting, standing, and laying).

### Key Features of the Dataset

- **Activities**: 6
- **Subjects**: 30
- **Features**: 561
- **Data**: Time and frequency domain variables

## Contents

The repository contains the following Jupyter notebooks:

### 1. Centralized Learning

- **Notebook Directory**: `CL`
- **Description**: This notebook demonstrates the implementation of a centralized learning model using `scikit-learn`. All data is aggregated on a central server for training.
- **Dependencies**: `scikit-learn`

### 2. Federated Learning

- **Notebook Directory**: `FL`
- **Description**: This notebook demonstrates the implementation of a federated learning model using the Flower framework. The training is distributed across multiple devices, and only model updates are shared with a central server.
- **Dependencies**: `flwr` ([flwr | link](https://flower.ai/))

## Usage

### Prerequisites

Ensure you have the following installed:

- Python 3.12.2
- Jupyter Notebook or JupyterLab
<!-- - Required Python packages (see [Requirements](#requirements))

### Requirements

Install the necessary Python packages using `pip`:

```bash
pip install -r requirements.txt -->