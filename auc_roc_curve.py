from sklearn.metrics import roc_curve, roc_auc_score
import numpy
import pandas as pd

#above 0.7 is acceptable

df = pd.read_csv(r"C:\Users\gohti\OneDrive - Nanyang Technological University\FYP_ROS_OOD\emptytrack_1_duckcutoff_awc1_final.csv")
ground_truth = df["ground_truth"].to_numpy()
y_score = df["test_results"].to_numpy()
r_auc = roc_auc_score(ground_truth,y_score)
print(r_auc)