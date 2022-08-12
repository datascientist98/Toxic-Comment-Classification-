import numpy as np 
import pandas as pd
import sys
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score


'''Function that returns f-score, accuracy
precision, recall for a multilabel classification '''

def evaluate(y_labels, pred_labels): 

    acc = accuracy_score(y_labels, pred_labels)
    fscore = f1_score(y_labels, pred_labels, average = 'weighted')
    precision = precision_score(y_labels, pred_labels,  average = "weighted")
    recall = recall_score(y_labels, pred_labels,  average = "weighted")

    return acc, fscore, precision, recall


no_of_args = len(sys.argv)
if no_of_args == 1:
    print("Please provide absolute path to true and predicted labels.")
    sys.exit()
elif no_of_args == 2:
    print('Only one path has been provided. Two are required to compute scores.')
    sys.exit()
elif no_of_args == 3:
    
    actual_df = pd.read_csv(sys.argv[1])
    pred_df = pd.read_csv(sys.argv[2])
else:
    print("Too many arguments!")
    sys.exit()

actual_labels = actual_df[["toxic", "severe_toxic", "obscene","threat","insult","identity_hate"]].to_numpy()
pred_labels =  pred_df[["toxic", "severe_toxic", "obscene","threat","insult","identity_hate"]].to_numpy()

evaluate(actual_labels, pred_labels)