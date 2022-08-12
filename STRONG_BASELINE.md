1. For the strong baseline, we decided to implement logistic regression with TFIDF vectorization and data preprocessing. 

2. The basic librries used are - numpy, pandas, sklearn. Install the required libraries using pip.

3. After mounting the google drive which contains our train, validation and test files, we loaded the above mentioned files into respective dataframes (for Google Colab users). To run from the command line, download the data from https://drive.google.com/drive/u/0/folders/1lXR7WkXRRSwX9YScdNaVvfWbBCzFXHCm and load the data using the pd.read_csv function in the code. The path to the data on your local system will be passsed as a command line argument. 

4. Next, we defined a function clean(str) to replace the '\n' and '\t' characters in the comment text with blank spaces as these characters do not contribute to the label. This function was vectorized in replace_invalid_chars(data).

5. There were some records which had '-1' as their label for one or more categories. On referring to the information about the data, we learnt that the '-1' label means that "the value of -1 indicates it was not used for scoring". Therefore, we dropped the rows from the data which had a label of -1.

6. We checked for missing values in the data using isna() and found that there are no missing values in the data.

7. The max length of text in training data was found to be 5000. We calculated this inorder to set the parameters of the TFIDF vectorizer (sklearn.feature_extraction.text). Using this, we transformed the training, validation and testing data. 

8. Using the cross_val_score, we took one label at a time and used the text to perform Logistic Regression (sklearn.liner_model). 

9. Then we calculated the cv score and stored it in a list. Finally, we calculated the total cv score by computing the mean of the scores obtained per label. The predicted labels are converted from a dataframe to a csv so that it can be passed to the evaluation script.

10. We also calculated the f1 score and confusion matrix per label using sklearn.metrics in order to understand the classification results.


