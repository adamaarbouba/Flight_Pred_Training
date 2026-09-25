# Flight Price Prediction

This project is a machine learning and data analysis project for predicting airline ticket prices based on flight attributes such as airline, departure time, route, number of stops, class, duration, and how many days remain before departure.

## Overview

The dataset in [core/flights.csv](core/flights.csv) contains flight booking data and is used to train a regression model that estimates the target variable: `price`.

The workflow includes:
- data exploration and validation
- preprocessing and feature engineering
- one-hot encoding of categorical features
- scaling of selected numeric variables
- model training and evaluation
- saving trained models for reuse

## Project structure

- [core/flights.csv](core/flights.csv) — raw flight dataset used for training
- [notebook/check_data.ipynb](notebook/check_data.ipynb) — exploratory data analysis and data quality checks
- [notebook/preprocessing.ipynb](notebook/preprocessing.ipynb) — preprocessing, encoding, and feature preparation
- [notebook/model.ipynb](notebook/model.ipynb) — train/test split, model training, and evaluation
- [models/LR_model.joblib](models/LR_model.joblib) — trained Linear Regression model
- [models/modeldumb.joblib](models/modeldumb.joblib) — baseline regression model for comparison

## Dataset fields

The dataset includes columns such as:
- `airline`
- `flight`
- `source_city`
- `departure_time`
- `stops`
- `arrival_time`
- `destination_city`
- `class`
- `duration`
- `days_left`
- `price`

The target variable is `price`, and the other columns are used as predictors.

## Data preparation

The preprocessing notebook performs the main transformations used for modeling:
- loads the CSV into a pandas DataFrame
- checks for missing or inconsistent values
- converts categorical values into encoded features
- normalizes selected numeric features like `duration` and `days_left`
- removes unnecessary fields before model training

## Model training

The modeling notebook uses a supervised regression approach. The project includes:
- a baseline `DummyRegressor` model for comparison
- a `LinearRegression` model trained on encoded and scaled features
- evaluation using a train/test split and model scoring

## How to use

1. Activate the environment:
   ```bash
   source ~/.venvs/jupyter/bin/activate
   ```
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the notebooks in this order:
   - `check_data.ipynb`
   - `preprocessing.ipynb`
   - `model.ipynb`

## Notes

This repository is focused on the data science pipeline rather than a web application or API. It is designed to demonstrate how flight fare prediction can be built from historical flight data using common preprocessing and regression techniques.
