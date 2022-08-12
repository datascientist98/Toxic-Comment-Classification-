import numpy as np 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import sys
import pandas as pd


def evaluate(y_labels, pred_labels): 

    acc = accuracy_score(y_labels, pred_labels)
    fscore = f1_score(y_labels, pred_labels)
    precision = precision_score(y_labels, pred_labels)
    recall = recall_score(y_labels, pred_labels)

    return acc, fscore, precision, recall


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Missing arguments')
        
    else:
        actual_df = pd.read_csv(sys.argv[1])
        pred_df = pd.read_csv(sys.argv[2])
        
        acc, fscore, precision, recall = evaluate(actual_df, pred_df)
        
        print('Accuracy =', acc)
        print('F-Score =', fscore)
        print('Precision = ', precision)
        print('Recall = ', recall)