# mlops_kubeflow
Exploring ML pipelines with kubeflow:

The focus will be more on using the machine learning pipelines, rather than the purely data science model training side of it.

Business problem: identification (and hence preventing) fraudulent credit card transactions.

Type of ML problem: Classification.

No. of classes : 2

Dataset source: https://www.kaggle.com/c/ieee-fraud-detection/data

"The data is broken into two files identity and transaction, which are joined by TransactionID." 
Hence, inner join was applied to join the following two tables/files:
- train_identity
- train_transaction

