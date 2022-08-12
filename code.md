
# simple_baseline.py

For our weak baseline, we used a term frequency inverse document frequency vectorizor the text from the comments in order to extract our features. We fed these features into the sklearn's DummyClassifier, which is a classifier that makes predictions using simple rules. We used a "stratified" approach, which generates predictions by respecting the training set's class distribution. Stratified sampling aims to split the members of the population into different subpopulations based on other samples that are similar, so this is often better than other approaches. Other approaches might be like comparing apples to oranges. We fitted separate dummy classifiers for each label, so the classifier lacks knowledge of the other labels, which may be a disadvantage. This is particularly important in the case of knowing whether or not the comment was classified as toxic, since all other labels are false if the comment is not labeled as toxic. The lack of complexity and knowledge in the model make it a good simple baseline.

The simple baseline scored on the testing data as follows:

|-----------|---------------------|
| Accuracy  | 0.7208149353164459  |
|-----------|---------------------|
| F1 Score  | 0.06471058222778961 |
|-----------|---------------------|
| Precision | 0.0651262885453231  |
|-----------|---------------------|
| Recall    | 0.06434962147281487 |
|-----------|---------------------|

Overall, this is a relatively high accuracy, considering we are performing binary classification, but much of this relates to class imbalance. There are far more comments that are not toxic than there are toxic, and of the comments that are toxic, it is much more common that toxic comments express 1 or 2 types of toxicity as opposed to all types. As a result, we have many more 0 labels than 1 labels, and the model is biased towards these predictions, which means it is easier to be accurate in doing so. The F Score, Precision, and Recall were low in relation to the Accuracy because even though the predictions were typically accurate, there were many false positives and false negatives.

In order to run this file to make predictions, run the following command: 

```python simple_baseline.py [train_data].csv [test_data].csv [output_path].csv```

such that:
* [train_data] is the file path of the training data
* [test_data] is the file path of the data to predict labels for
* [output_path] is the file path to output the predictions to

# strong_baseline.py

For the strong baseline, we decided to implement logistic regression with TFIDF vectorization and data preprocessing. The primary python packages we utilized used are - numpy, pandas, sklearn. We installed the required libraries using pip.

Next, we defined a function clean(str) to replace the '\n' and '\t' characters in the comment text with blank spaces as these characters do not contribute to the label. This function was vectorized in replace_invalid_chars(data). There were some records which had '-1' as their label for one or more categories. On referring to the information about the data, we learnt that the '-1' label means that "the value of -1 indicates it was not used for scoring". Therefore, we dropped the rows from the data which had a label of -1. We checked for missing values in the data using isna() and found that there are no missing values in the data.

The max length of text in training data was found to be 5000. We calculated this inorder to set the parameters of the TFIDF vectorizer (sklearn.feature_extraction.text). Using this, we transformed the training, validation and testing data. Using the cross_val_score, we took one label at a time and used the text to perform Logistic Regression (sklearn.linear_model). 

The strong baseline attained the following scores:

|-----------|---------|
| Accuracy  | 0.90921 |
|-----------|---------|
| F1 Score  | 0.54734 |
|-----------|---------|
| Precision | 0.80240 |
|-----------|---------|
| Recall    | 0.42567 |
|-----------|---------|

In order to run this file to make predictions, run the following command: 

```python strong_baseline.py [train_data].csv [val_data].csv [test_data].csv [output_path].csv```

such that:
* [train_data] is the file path of the training data
* [val_data] is the file path of the validation data
* [test_data] is the file path of the data to predict labels for
* [output_path] is the file path to output the predictions to

# logistic_regression_with_preprocessing.py

This is an extension of strong_baseline.py. In order to improve the model, we incorporated further preprocessing.

In order to run this file to make predictions, run the following command: 

```python logistic_regression_with_preprocessing.py [train_data].csv [val_data.].csv [test_data].csv [output_path].csv```

such that:
* [train_data] is the file path of the training data
* [val_data] is the file path of the validation data
* [test_data] is the file path of the data to predict labels for
* [output_path] is the file path to output the predictions to

# logistic_regression_with_preprocessing.py

This is an extension of strong_baseline.py. In order to improve the model, we incorporated further preprocessing.

In this extension we implemented more significant data preprocessing by making the following improvements:

1. Removal of stopwords
2. Cleaning text by replacing invalid characters
3. Removing the column 'Unnamed: 0' as it does not contribute to the label
4. Dropping the rows which had '-1' in the label

Using seaborn, we plotted the number of records in each label and observed that there was a huge class imbalance between the non-toxic class as compared to the other classes of toxicity in the label. 

For analysis, we decided to run the same Logistic Regression model for binary classification. We added an additional column 'BinaryLabel' which would be 0 if ALL the labels for the comment were 0 and 1 otherwise.

Conclusion: Due to sparsity in labels of toxicity and its types as compared to the non-toxic text, the binary classification model performs better than the multilabel multiclass classification problem posed above in terms of f1 score and confusion matrix both.

Next, we implemented LSTM on the multiclass, multilabel dataset. 
The libraries used is - keras. From keras, we used the following: Model, Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, text, sequence.

We initialize the keras.text.Tokenizer and fit it on the training comment_text to convert it into its tokenized representation (sequence of integers). We used paddding to transform all input sequences into sequences whose length is equal to that of the largest sequence. The get_model function specifies the achitecture of model that we will be using for our task. model.fit will train the model and then it can be saved and loaded using the functions model.save and model.load respective

In order to run this file to make predictions, run the following command: 

```python logistic_regression_with_preprocessing.py [train_data].csv [val_data.].csv [test_data].csv [output_path].csv```

such that:
* [train_data] is the file path of the training data
* [val_data] is the file path of the validation data
* [test_data] is the file path of the data to predict labels for
* [output_path] is the file path to output the predictions to

# binary_logistic_regression.py

This is an extension of logistic_regression_with_preprocessing.py. We decided there might be utility in compounding the labels to form overall binary labels for training and predictions, simplifying our problem and potentionally improving our results.

In order to run this file to make predictions, run the following command: 

```python binary_logistic_regression.py [train_data].csv [val_data.].csv [test_data].csv [output_path].csv```

such that:
* [train_data] is the file path of the training data
* [val_data] is the file path of the validation data
* [test_data] is the file path of the data to predict labels for
* [output_path] is the file path to output the predictions to


# binary_lstm.py

After we began to perceive that we were approaching the limits of the performance of the Logistic Regression, we speculated that we might achieve better performance with a higher complexity model. We chose to implement LSTM due to the fact that LSTM has the ability to identify long term dependencies to a degree that other models cannot.

We tokenized the words and transformed our sentences to sequences of integers to represent our features. Next, we applied padding so that all sequences contained the length of the longest sentence.

After cross validating, and training different LSTM models varying dropout, batch size, and units, we arrived at our best model, which had the following:

* batch size of 32
* dropout of 0.25
* units of 8

We attained the following results:

|-----------|--------|
| Accuracy  | 0.9401 |
|-----------|--------|
| F1 Score  | 0.6625 |
|-----------|--------|
| Precision | 0.695  |
|-----------|--------|
| Recall    | 0.6325 |
|-----------|--------|

In order to run this file to make predictions, run the following command: 

```python binary_lstm.py [train_data].csv [val_data.].csv [test_data].csv [output_path].csv```

such that:
* [train_data] is the file path of the training data
* [val_data] is the file path of the validation data
* [test_data] is the file path of the data to predict labels for
* [output_path] is the file path to output the predictions to


# utils.py

Packages and functions across the various models.