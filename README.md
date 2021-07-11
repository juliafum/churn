# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, you identify credit card customers that are most likely to churn. The projectincludes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).
The project has the following structure:
```bash
├── Guide.ipynb
├── README.md
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── churn_distribution.png
│   │   ├── customer_age_distribution.png
│   │   ├── heatmap.png
│   │   ├── martial_status_distribution.png
│   │   └── total_transaction_distribution.png
│   └── results
│       ├── feature_importance.png
│       ├── logistics_results.png
│       └── rf_results.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
└── requieremts.txt
```
The trained models are stored in models. In images you can find plots from EDA and model validation.

## Running Files
Install dependencies with 
```bash
pip3 install -r requirements.txt
```
Run training and prediction with
```bash
python churn_library.py
```
Run tests with
```bash
python churn_script_logging_and_tests.py
```
In train and predict mode (DO_TRAIN=1), the model is trained and saved as pkl file. Next predition is done on test data.
In predict mode (DO_TRAIN=0), the model object is loaded and applied to test data.


