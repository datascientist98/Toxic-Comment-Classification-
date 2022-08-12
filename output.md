# Scoring
In our evaluation script, we use f_score, accuracy, precision, and recall to evaluate our methods.

## System Requirements
The following packages are required in order to run the script
- numpy
- pandas
- sys
- sklearn.metrics: f_score, accuracy_score, precision_score, recall_score

# Metrics
- precision: True Positives/ True Positives + False Positives
- recall: True Positive/ True Positive + False Negative
- f_score: 2 x precision x recall/(precision+recall)
- accuracy_score: True Positive + True Negative/ Total Predictions


## Steps to Run Script
1. Pass the absolute path of the actual labels csv and predicted labels csv
    - Example:
    python3 score.py 'path/to/test/' 'path/to/predict'

2. Evaluate function: This function takes in the set of true labels, and predicted labels and computes the accuracy, fscore, precision and recall 
    ```
    def evaluate(y_labels, pred_labels): 
    ```
    - Example input 1: [0,1,1], [0,0,0]

        Output: (0.3333333333333333, 0.0, 0.0, 0.0)
    - Example Input 2: [1,1,1], [0,0,0]
        Output: (0.0, 0.0, 0.0, 0.0)

    - Example Input 3: [0,1,1,1,1,1], [1,1,0,1,1,0]

        Output: (0.5, 0.6666666666666665, 0.75, 0.6)