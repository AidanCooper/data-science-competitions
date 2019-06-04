# Competition Notebooks

Examples of data preparation and predictive modelling for data science competitions on Kaggle/similar covering a range of classification, regression and forecasting tasks for various data types and machine learning techniques.

---

## [CareerCon 2019](CareerCon2019/CareerCon2019.ipynb)
Jupyter notebook for the CareerCon 2019 [Kaggle competition](https://www.kaggle.com/c/career-con-2019).

**Techniques implemented in this notebook:**
- Classification of multivariate time series
- Signal processing and linking of continuous data
- Exploiting data leakage between test and training sets
- Random forest classifier, with hyperparameter optimisation via cross validation and grid search.

---

## [TMDB Box Office Prediction](BoxOfficePrediction/CatBoost-CatBoost_Encoding-Additional_Data.ipynb)
Jupyter notebook for the TMDB Box Office Prediction [Kaggle competition](https://www.kaggle.com/c/tmdb-box-office-prediction/).

**Techniques implemented:**
- Gradient Boosted Regression Tree using CatBoost.
- Multi-label target encoding of categorical data with high-cardinality.

---

## [House Prices: Advanced Regression Techniques](HousePricesAdvancedRegression/HousePricesLinearModel.ipynb)
Jupyter notebook for the House Prices: Advanced Regression Techniques [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Techniques implemented in this notebook:**
- Feature processing to give normalised distributions for improved linear model performance.
- Outlier removal from numerical features.
- Feature engineering and feature selection. Data preprocessing: imputation of missing values; categorical feature encoding.
- Pipeline for data processing.
- Lasso regression model, with hyperparameter optimisation via cross validation and grid search.

---

## [Don't Overfit! II](DontOverfit/DontOverfit.ipynb)
Jupyter notebook for the Don't Overfit! II [Kaggle competition](https://www.kaggle.com/c/dont-overfit-ii).

**Techniques implemented in this notebook:**
- Adding noise to the small training data set to reduce overfitting.
- Recursive feature elimination with cross-validation (RFECV).
- Generating multiple models using 20-fold stratified shuffle split and cross validation, and ensembling those that met a minimum score threshold.
- Ensembling of lasso, ridge and support vector regression models.

---

## [LANL Earthquake Prediction](EarthquakePrediction/020-model-GBM.ipynb)
Jupyter notebook for the LANL Earthquake Prediction [Kaggle competition](https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview).

**Techniques implemented in this notebook:**
- Feature engineering/signal processing of continuous acoustic measurements.
- Gradient boosted regression tree models using the CatBoost and LightGBM libraries.

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
