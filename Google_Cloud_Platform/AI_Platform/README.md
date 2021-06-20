# mlops_kubeflow
Step 1: Run the credit_card_fraud.ipynb notebook 
Step 2: Run the kfp notebook.

In Step 1:
- we do some data pre-processing
- build a XGBoost classification model
- containerize (Dockerize) the training script and push the container-image to Google Container Registry (GCR) 
- and finally do some hyper-parameter tuning

Note: Feature selection has already been done using the feature importance functionality of AutoML Tables.
Hence, we already know which columns to pick, for our custom model training.

In Step 2, we build a kubeflow pipeline, using custom (data ingestion) as well as prebuilt (model training and deployment) components.

Train-Validation split set as 80:20

Hyper-parameter tuning has been turned off, since that has already been done earlier.
Hence, now using (rather hard-coding) the parameter combo which yielded best model performance.

For those familiar with Airflow, the pipeline compilation here is actually creating a DAG with each kubeflow component being used as an operator to create a task in the DAG.

Shoutout to Kyle Steckler from Google ASL team, who was the mentor for this project.