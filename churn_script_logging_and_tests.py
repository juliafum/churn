# pylint: disable=W0621, R0913, C0103
'''
Testing & Logging of the Churn prediciton library 
author: Julia
date: July 10, 2021
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import 
    input:
                        import_data: method to import data

    output: 
                        None
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    test EDA artifacts available
    input:
                        perform_eda: method to perform eda

    output: 
                        None


    '''
    df = cls.import_data("./data/bank_data.csv")
    try:
        df = perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except NameError as err:
        logging.error("Testing perform_eda: df is not definied")
        raise err

    try:
        assert os.path.isfile("./images/eda/churn_distribution.png")
        assert os.path.isfile("./images/eda/customer_age_distribution.png")
        assert os.path.isfile("./images/eda/heatmap.png")
        assert os.path.isfile(
            "./images/eda/martial_status_distribution.png")
        assert os.path.isfile(
            "./images/eda/total_transaction_distribution.png")
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing permorm_ede: The file doesn't exists")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    test df has columns with _Churn suffix
    input:
                        perform_eda: method to encode variables

    output: 
                        None
    '''
    df = cls.import_data("./data/bank_data.csv")
    df = cls.perform_eda(df)
    category_lst = ["Gender", "Education_Level",
                    "Marital_Status", "Income_Category", "Card_Category"]
    try:
        df = encoder_helper(df, category_lst)
        logging.info("Testing encoder_helper: SUCCESS")
    except NameError as err:
        logging.error("Testing encoder_helper: df is not definied")
        raise err

    try:
        assert df.columns.str.contains("_Churn").any()
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: df doesn't  have columns with _Churn suffix")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    input:
                        perform_feature_engineering: method to feature engineering

    output: 
                        None


    '''
    df = cls.import_data("./data/bank_data.csv")
    df = cls.perform_eda(df)
    category_lst = ["Gender", "Education_Level",
                    "Marital_Status", "Income_Category", "Card_Category"]

    df = cls.encoder_helper(df, category_lst)
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except NameError as err:
        logging.error(
            "Testing perform_feature_engineering: df is not definied")
        raise err

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Outputs don't appear to have rows")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    input:
                        train_models: method to train models

    output: 
                        None

    '''
    df = cls.import_data("./data/bank_data.csv")
    df = cls.perform_eda(df)
    category_lst = ["Gender", "Education_Level",
                    "Marital_Status", "Income_Category", "Card_Category"]

    df = cls.encoder_helper(df, category_lst)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df)

    try:
        train_models(X_train, X_test, y_train, y_test, DO_TRAIN=0)
        logging.info("Testing train_model: SUCCESS")
    except NameError as err:
        logging.error(
            "Testing train_model: Input is not definied")
        raise err

    try:
        assert os.path.isfile("./models/logistic_model.pkl")
        assert os.path.isfile("./models/rfc_model.pkl")
    except AssertionError as err:
        logging.error(
            "Testing train_model: The file doesn't exists")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
