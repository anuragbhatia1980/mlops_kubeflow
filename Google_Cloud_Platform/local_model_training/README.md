# mlops_kubeflow

Type of dataset: imbalanced (as expected?)

How did we handle class imbalance?
- choice of performance metric:
a) Accuracy is not appropriate. b) Recall matters more than Precision. Assumption: It is more important to minimize False-Negatives (i.e. allowing fraudulent transactions) than False-Positives (i.e. declining genuine transactions). c) ROC-AUC can also be considered.
- minority-oversampling
- majority-undersampling  
- hyper-parameter value during model training
  (e.g. 'scale_pos_weight' in case of XGBoost)
