import argparse
import hypertune

import numpy as np
import pandas as pd

import xgboost
from xgboost import XGBClassifier
import sklearn
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import SMOTENC


def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--learning_rate',
                        required=True,
                        type=float,
                        help='learning rate')
    parser.add_argument(
                        '--max_depth',
                        required=True,
                        type=int,
                        help='maximum depth of boosted tree')
    parser.add_argument(
                        '--scale_pos_weight',
                        required=True,
                        type=int,
                        help='weights ratio of negative to positive classes')

    args = parser.parse_args()
    return args


def build_model(bucket_uri,  # GCS bucket path
                learning_rate,
                max_depth,
                scale_pos_weight
                ):
    
    # Get training and test datasets (from GCS bucket)
    training_dataset_path = bucket_uri + "train.parquet"
    test_dataset_path = bucket_uri + "eval.parquet"
    
    df_train = pd.read_parquet(training_dataset_path)
    df_test = pd.read_parquet(test_dataset_path)
    
    numeric_feature_indexes = list(range(0,15))
    categorical_feature_indexes = list(range(15,18))

    preprocessor = ColumnTransformer(
                                     transformers=[
                                                   ('num', StandardScaler(), numeric_feature_indexes),
                                                   ('cat', OneHotEncoder(), categorical_feature_indexes) 
                                                  ]
                                    )

    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', XGBClassifier(learning_rate=learning_rate,
                                                     max_depth=max_depth,
                                                     scale_pos_weight=scale_pos_weight                       
                                                    ))
                        ])
    
    num_features_type_map = {feature: 'float64' for feature in df_train.columns[numeric_feature_indexes]}
    df_train = df_train.astype(num_features_type_map)
    df_test = df_test.astype(num_features_type_map)
    
    # X:y splits, for training and test
    y_train = df_train['isFraud'] # target/label column
    X_train = df_train.drop('isFraud', 
                            axis=1, # columns
                            inplace=False) # The original dataframe remains the same
    
    y_test = df_test['isFraud'] # target/label column
    X_test = df_test.drop('isFraud', 
                          axis=1, # columns
                          inplace=False) # The original dataframe remains the same
    
    
    X_train, y_train = SMOTENC(categorical_features=categorical_feature_indexes).fit_resample(X_train, 
                                                                                              np.array(y_train))
    
    return pipeline, X_train, y_train, X_test, y_test


def main():
    args = get_args()

    bucket_uri = "gs://kubeflow-1-0-2-kubeflowpipelines-default/card_fraud_data/"  # <-- Change this
    pipeline, X_train, y_train, X_test, y_test = build_model(
                                                             bucket_uri,  
                                                             args.learning_rate,
                                                             args.max_depth,
                                                             args.scale_pos_weight
                                                            )

    pipeline.fit(X_train, y_train)

    # y_pred = pipeline.predict(X_test)
    predictions = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test,
                            predictions)

    # DEFINE METRIC
    hp_metric = roc_auc  # metric to be optimized

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
                                            hyperparameter_metric_tag='roc_auc',
                                            metric_value=hp_metric,
#                                             global_step=10  # no. of epochs
                                            )

if __name__ == "__main__":
    main()