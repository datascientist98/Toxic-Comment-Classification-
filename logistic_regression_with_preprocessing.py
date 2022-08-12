from sklearn.linear_model import LogisticRegression

from utils import *

if __name__ == '__main__': 

    try: 
        df_train = pd.read_csv(sys.argv[1])
        df_val = pd.read_csv(sys.argv[2])
        df_test = pd.read_csv(sys.argv[3])
        
        test_label_output_path = sys.argv[4]

    except Exception as ex:
        print(f'Missing argument(s). {ex}')
        sys.exit(1)

    stopword_list=STOP_WORDS
        
    df_train = cleanup_df(df_train)
    df_val = cleanup_df(df_val)
    df_test = cleanup_df(df_test)

    lens = df_train.comment_text.str.len()
    lens.mean(), lens.std(), lens.max()

    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    df_train_text = df_train['comment_text']
    df_val_text = df_val['comment_text']
    df_test_text = df_test['comment_text']

    word_vectorizer.fit(df_train_text)

    train_features = word_vectorizer.transform(df_train_text)
    val_features = word_vectorizer.transform(df_val_text)
    test_features = word_vectorizer.transform(df_test_text)

    scores = []
    val_output = pd.DataFrame.from_dict({'id': df_val['id']})
    test_output = pd.DataFrame.from_dict({'id': df_test['id']})

    for label in labels:

        train_target = df_train[label]
        clf = LogisticRegression(C=0.1, solver='sag')
        cv_score = np.mean(cross_val_score(clf, train_features, train_target, cv=3))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(label, cv_score))

        clf.fit(train_features, train_target)
        val_output[label] = clf.predict(val_features)
        test_output[label] = clf.predict(test_features)
        
    print('Total CV score is {}'.format(np.mean(scores)))

    print("For validation data -")
    for label in labels:
      fs_val = f1_score(val_output[label], df_val[label])
      print("f1_score of label",label,"is: ",fs_val)
    print("For test data -")
    for label in labels:
      fs_test = f1_score(test_output[label], df_test[label])
      print("f1_score of label",label,"is: ",fs_test)

    print("For validation data -")
    for label in labels:
      cm_val = confusion_matrix(df_val[label], val_output[label])
      print(label,'\n',cm_val)
    print("For test data -")
    for label in labels:
      
      cm_test = confusion_matrix(df_test[label], test_output[label])
      print(label, '\n', cm_test)

    ncols = len(labels)
    fig, axes = plt.subplots(1, ncols, figsize = (30,5))
    for label, ax in zip(labels, axes.flatten()):
      axes[0] = sns.countplot(x = label, data=df_train, orient = 'v', ax=ax)
      axes[1] = sns.countplot(x = label, data=df_train, orient = 'v', ax=ax)
      axes[2] = sns.countplot(x = label, data=df_train, orient = 'v', ax=ax)
      axes[3] = sns.countplot(x = label, data=df_train, orient = 'v', ax=ax)
      axes[4] = sns.countplot(x = label, data=df_train, orient = 'v', ax=ax)
      axes[5] = sns.countplot(x = label, data=df_train, orient = 'v', ax=ax)
      
    plt.show()

    test_output.to_csv(test_label_output_path ,index=False)
