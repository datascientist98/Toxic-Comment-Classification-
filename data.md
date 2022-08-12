Markdownfile of Data - 

The datset comes from the Kaggle Toxic Comment Challenge (2017). The files of the dataset are in .csv format. 
Link to the original dataset - 
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

The challenge of the competition was to identify and classify toxic online comments. The dataset has been annotated by human raters. There are 6 categories of toxicity as defined in the dataset:

1. Toxic : The definiton of toxicity provided in the competition is - "comments that are rude, disrespectful or otherwise likely to make someone leave a discussion"
NOTE - The competiton has not defined strict criteria for the meaning of the rest of the labels present in the dataset. However, we are assuming a general intuitive meaning of the labels among the raters based on dictionary definitions.
2. Severe_Toxic 
3. Obscene 
4. Threat
5. Insult
6. Identity_Hate

Description of the data given in the competition (original dataset):
1. Training_data.csv-
    Number of training samples = 159572
    Features - ID of user, comment text
    Labels - Toxic, Severe_Toxic, Obscene, Threat, Insult, Identity_Hate
    The given labels are binary - can take 2 values, i.e. 0 or 1. The label is 0 if the text does not belong to that class and the label is 1 if the text does belong to that class. 

2. Testing_data.csv-
    Number of training samples = 153165
    Features - ID of user, comment text

For our project, we will be combining the training and testing datsets (total number of samples = 312735) and dividing it into 3 separate files - train.csv (80% of the total data = 250190), validation.csv(10% of the total data = 31274) and test.csv(10% of the total data = 31271).

Link to the split dataset(Google drive) - 
https://drive.google.com/drive/u/0/folders/1lXR7WkXRRSwX9YScdNaVvfWbBCzFXHCm

This is a multi-label classification problem which means that one text can have multiple labels. Therefore, the labels are not mutually exclusive. 
For example, a text from the actual dataset - "Stupid peace of ..." has been classified as "Toxic", "Severe Toxic" and "Insult". Another intuitive insight from this is that some words might be misspelled (peace, when the intended usage in this case is piece) and therefore, preprocessing will be required.  

