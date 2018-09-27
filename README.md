# Kaggle Competition Notebooks

Examples of data preparation and predictive modelling for Kaggle competitions covering a range of classification, regression and forecasting tasks for various data types and machine learning techniques.

---

## [House Prices: Advanced Regression Techniques](HousePricesAdvancedRegression/HousePricesLinearModel.ipynb)
Jupyter notebook for the House Prices: Advanced Regression Techniques [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Techniques implemented in this notebook:**
- Feature processing to give normalised distributions for improved linear model performance.
- Outlier removal from numerical features.
- Feature engineering and feature selection.
- Imputation of missing values.
- Categorical feature encoding.
- Pipeline for data processing.
- Lasso regression model, with hyperparameter optimisation via cross validation and grid search.

---

## [Plant Seedlings Classification](PlantSeedlingsClassification)
Report detailing the use of Google Cloud AutoML Vision to rapidly build a multinomial image recognition model for the Plant Seedlings Classification [Kaggle competition](https://www.kaggle.com/c/plant-seedlings-classification).

**Techniques implemented in this report:**
- Preparation and loading of image data into AutoML's web interface.
- Model training and overview of results (precision, recall, f1 score, confusion matrix).
- Jupyter notebook for a Python script that enables bulk predictions to be made via the AutoML Vision API.

---

## [Store Item Demand Forecasting Challenge](StoreItemDemand)
A collection of Jupyter notebooks for the Store Item Demand Forecasting Challenge [Kaggle competition](https://www.kaggle.com/c/demand-forecasting-kernels-only).

**Techniques implemented in this notebook:**
- Simple time series forecasting performance benchmark models (Average Method, Seasonal Naive Method).
- Facebook Prophet model.
- ARIMA, ARIMAX and SARIMA models.
- Time series decomposition analysis and Dickey-Fuller testing.
- (Seasonal) data differencing.
- Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) analysis.

---

## [Merck Molecular Activity Challenge](MolecularActivity/MolecularActivity.ipynb)
Jupyter notebook for the Merck Molecular Activity Challenge [Kaggle competition](https://www.kaggle.com/c/MerckActivity).

**Techniques implemented in this notebook:**
- Simple approach to handling and processing fifteen large datasets separately but efficiently.
- Automated feature selection based on information content of features.
- Random forest regressor, with hyperparameter optimisation via cross validation and grid search.

---
