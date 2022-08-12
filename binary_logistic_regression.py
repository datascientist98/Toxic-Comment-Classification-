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
        
    df_train = cleanup_df(df_train)
    df_val = cleanup_df(df_val)
    df_test = cleanup_df(df_test)
       
    df_train_binary = get_binarized_df(df_train)
    df_val_binary = get_binarized_df(df_val)
    df_test_binary = get_binarized_df(df_test)

    df_train_text_binary = df_train_binary['comment_text']
    df_val_text_binary = df_val_binary['comment_text']
    df_test_text_binary = df_test_binary['comment_text']

    word_vectorizer.fit(df_train_text_binary)

    train_features_binary = word_vectorizer.transform(df_train_text_binary)
    val_features_binary = word_vectorizer.transform(df_val_text_binary)
    test_features_binary = word_vectorizer.transform(df_test_text_binary)

    scores = []
    val_output_binary = pd.DataFrame.from_dict({'id': df_val_binary['id']})
    test_output_binary = pd.DataFrame.from_dict({'id': df_test_binary['id']})

    train_target = df_train_binary['BinaryLabel']
    clf = LogisticRegression(C=0.1, solver='sag')

    clf.fit(train_features_binary, train_target)
    val_output_binary['BinaryLabel'] = clf.predict(val_features_binary)
    test_output_binary['BinaryLabel'] = clf.predict(test_features_binary)

    test_output_binary.to_csv(test_label_output_path, index=False)

    print("For validation data -")
    fs_val_binary = f1_score(val_output_binary['BinaryLabel'], df_val['BinaryLabel'])
    print("f1_score of binary classification is: ",fs_val_binary)

    print("For test data -")
    fs_test_binary = f1_score(test_output_binary['BinaryLabel'], df_test_binary['BinaryLabel'])
    print("f1_score of binary classification is: ",fs_test_binary)

    test_acc = accuracy_score(test_output_binary['BinaryLabel'], df_test['BinaryLabel'])
    val_acc = accuracy_score(test_output_binary['BinaryLabel'], df_test['BinaryLabel'])
    print(test_acc, val_acc)