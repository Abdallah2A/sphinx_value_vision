# Sphinx Value Vision

<p align="center">
  <img src="logo.png" alt="Sphinx Value Vision Logo" width="180" align="right">
</p>

Sphinx Value Vision is a full-cycle machine learning project for predicting house prices using an Egyptian dataset. The project leverages web scraping, data cleaning, visualization, and neural network model training with PyTorch. An API is provided for real-time predictions and the entire solution is containerized using Docker.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Model Training](#model-training)
5. [Pipeline (ZenML)](#pipeline-zenml)
6. [API](#api)
7. [Docker](#docker)
8. [Evaluation Metrics & Correlation Matrix](#evaluation-metrics--correlation-matrix)
9. [How the Model Works](#how-the-model-works)
10. [Setup & Usage](#setup--usage)
11. [License](#license)
12. [References](#references)

---

## Overview

**Sphinx Value Vision** predicts house prices from an Egyptian real estate dataset by executing a full machine learning lifecycle. The project includes:

- **Data Collection:** A Scrapy spider scrapes real estate listings.
- **Data Cleaning:** The dataset is processed to remove noise and outliers.
- **Data Visualization:** Histograms, scatter plots, box plots, and a correlation matrix are generated for exploratory analysis.
- **Model Training:** A neural network regression model is built with PyTorch.
- **Deployment:** An API serves the model predictions and Docker ensures reproducible deployment.

---

## Project Structure

```bash
sphinx_value_vision/
├── data/                     # Raw and processed data
│   ├── dataset.csv              # processed dataset
│   └── raw_dataset.csv              # raw dataset
├── model/
│   ├── sphinx_value_vision.pt
│   └── sphinx_value_vision.pkl
├── src/
│   ├── api/
│   │   ├── client.py                 # Example client for testing the FastAPI service
│   │   └── main.py                   # Main entry point (FastAPI app)
│   ├── data/
│   │   └── sphinx_value_vision/
│   │   │   ├── sphinx_value_vision/
│   │   │   │   ├── spiders/
│   │   │   │   │   └── aqarmap_spider.py
│   │   │   │   ├── items.py
│   │   │   │   ├── middlewares.py
│   │   │   │   ├── pipelines.py
│   │   │   │   └── settings.py
│   │   │   └── scrapy.cfg
├── ├── steps/
│   │   ├── clean_data_step.py
│   │   ├── collect_data_step.py
│   │   ├── train_model_step.py
│   │   └── visualize_dataset_step.py 
├── └── pipelines/
│       └── training_pipeline.py 
├── .dockerignore
├── Dockerfile
├── requirements.txt
└── run_pipeline.py           # Script to run the ZenML pipeline
```

---

## Dataset

The dataset consists of scraped housing data from a popular Egyptian real estate website. It includes various features such as price, area, room count, bathrooms, location, and more. The raw dataset is cleaned and processed to prepare it for training.

---

## Model Training

1. **Preprocessing**: 
  The raw dataset is cleaned, outliers are removed, and numerical/categorical features are prepared.

2. **Visualization**: 
  Visual analysis is performed using histograms, scatter plots, and box plots. A **Correlation Matrix** is also generated to explore feature relationships.

3. **Training**: 
  A neural network regression model is trained using PyTorch. The training process includes early stopping to avoid overfitting. Evaluation metrics like RMSE, MAE, and R² are computed.

4. **Artifacts**: 
  The trained model and preprocessor are saved for later deployment.

<p align="center">
  <img src="visual_matrices\correlation_matrix.png" alt="correlation matrix" width="400">
</p>

---

## Pipeline (ZenML)

The entire workflow is orchestrated using ZenML pipelines, which manage the sequential execution of:

- Data Collection (Scrapy)
- Data Cleaning
- Data Visualization
- Model Training

Run the pipeline using:

```bash
python training_pipeline.py
```

---

## API

An API is provided to serve the trained model for real-time house price prediction. You can send HTTP requests to the API endpoints to receive predictions.

Start the API with:

```bash
uvicorn src.api.main:app --reload
```
Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Docker

A Dockerfile is included for containerizing the project. To build and run the Docker container:

1. **Build the image:**
  ```bash
docker build -t sphinx-vision-value .
```
2. **Run the container:**
  ```bash
docker run -p 8000:8000 sphinx-vision-value
```
3. Access the API at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Evaluation Metrics

| Metric         | Value     |
|----------------|-----------|
| **RMSE**       | 459723.10 |
| **MAE**        | 300520.94 |
| **R² Score**   | 0.8321    |

We can find MAE is high, but it's normal because the prices is going up to 5 millions

---

## Setup & Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Abdallah2A/sphinx_value_vision.git
   ```
2. **Install Dependencies**:
   ```bash
   cd sphinx_value_vision
   pip install -r requirements.txt
   ```
3. **Configure ZenML** (if not already installed):
   ```bash
   zenml init
   ```
4. **Run the Pipeline**:
   ```bash
   python run_pipeline.py
   ```
5. **Start the FastAPI Service**:
   ```bash
   uvicorn src.api/main:app --reload
   ```
6. **Test the Inference**:
   - Use the provided `src/api/client.py` or any HTTP client (e.g., cURL, Postman) to send requests to `POST /predict`.

---

## License

This project is free to use.

---

## References

- [ZenML](https://docs.zenml.io/) - MLOps framework for building portable, production-ready MLOps pipelines.
- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast (high-performance) web framework for building APIs with Python.
- [Docker](https://www.docker.com/) - Container platform for packaging and deploying applications.

---

Enjoy using **Sphinx Value Vision** and happy house price predicting!
