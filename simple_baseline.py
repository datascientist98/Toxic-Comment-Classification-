from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
import sys
import pandas as pd
import numpy as np

def evaluate(y_labels, pred_labels): 

  return accuracy_score(y_labels, pred_labels, normalize = True)
  
  
if __name__ == '__main__':

    try: 
        train_df = pd.read_csv(sys.argv[1])
        test_df = pd.read_csv(sys.argv[2])
        
        test_label_output_path = sys.argv[3]
    
    except Exception as ex:
        print(f'Missing argument(s). {ex}')
        sys.exit(1)
        
    train_df_new = train_df[train_df.toxic != -1]

    comments = train_df_new['comment_text']
    Y_train_toxic = train_df_new['toxic']
    Y_train_severe_toxic = train_df_new['severe_toxic']
    Y_train_obscene = train_df_new['obscene']
    Y_train_threat = train_df_new['threat']
    Y_train_insult = train_df_new['insult']
    Y_train_identity_hate = train_df_new['identity_hate']

    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_model = vectorizer.fit(comments)
    X_train_tfidf = tfidf_model.transform(comments)
    X_tfidf_test = tfidf_model.transform(test_df["comment_text"])

    dummy_clf = DummyClassifier(strategy='stratified')

    dummy_clf.fit(X_train_tfidf, train_df_new["toxic"])
    predicts_toxic = dummy_clf.predict(X_tfidf_test)
    dummy_clf.fit(X_train_tfidf, train_df_new["severe_toxic"])
    predicts_sev_toxic = dummy_clf.predict(X_tfidf_test)
    dummy_clf.fit(X_train_tfidf, train_df_new["obscene"])
    predicts_obscene = dummy_clf.predict(X_tfidf_test)
    dummy_clf.fit(X_train_tfidf, train_df_new["threat"])
    predicts_threat = dummy_clf.predict(X_tfidf_test)
    dummy_clf.fit(X_train_tfidf, train_df_new["insult"])
    predicts_insult = dummy_clf.predict(X_tfidf_test)
    dummy_clf.fit(X_train_tfidf, train_df_new["identity_hate"])
    predicts_identity_hate = dummy_clf.predict(X_tfidf_test)
    
    results_df = pd.DataFrame(columns = ["toxic", "severe_toxic", "obscene","threat","insult","identity_hate"])
    
    results_df['toxic'] = predicts_toxic
    results_df['severe_toxic'] = predicts_sev_toxic
    results_df['obscene'] = predicts_obscene
    results_df['threat'] = predicts_threat
    results_df['insult'] = predicts_insult
    results_df['identity_hate'] = predicts_identity_hate
    
    results_df.to_csv(test_label_output_path, index=False)
    
    
    
    
    