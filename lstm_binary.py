from sklearn.multiclass import OneVsRestClassifier
from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from tensorflow.keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.backend import dropout
import os

from utils import *


def get_model_binary(dropout_val, units):

    embed_size = 32
    inp = Input(shape=(maxlen, ))
    x = Embedding(20001, embed_size)(inp)
    x = LSTM(units=units, return_sequences=True)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout_val)(x)
    x = Dense(5, activation="relu")(x)
    x = Dropout(dropout_val)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


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

    """Conclusion of Analysis: Due to sparsity in labels of toxicity and its types as compared to the non-toxic text, the binary classification model performs better than the multilabel multiclass classification problem posed above. The performance is in context to the F1 score. The confusion matrix also tells us that the model has predicted more labels correctly (TP and TN) as compared to the confusion matrices of multilabel multiclass outputs.

    To perform a comparison between two algorithms, we have built an LSTM model for our problem. The LSTM model will be using the multilabel multiclass data.
    """

    list_sentences_train = df_train['comment_text'].values
    list_sentences_test = df_test['comment_text'].values

    #initialize tokenizer
    tokenizer = text.Tokenizer(num_words=20000)

    tokenizer.fit_on_texts(list(list_sentences_train))

    #transform text to a sequence of integers
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

    #padding is done to transform all input sequences into sequences whose length is equal to that of the largest sequence
    maxlen = 2000
    X_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

    model = get_model()
    batch_size = 32
    epochs = 2

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_train = df_train[list_classes].values

    list_classes_binary = ["toxic"]
    y_train_binary = df_train[list_classes_binary].values

    # model_binary = get_model_binary(0.25, 4)
    # model_binary.fit(X_train, y_train_binary)

    # file_name = 'lstm_dropout0.25_units4.pth'
    # model_binary.save(file_name)

    # model_binary = get_model_binary(0.5, 4)
    # model_binary.fit(X_train, y_train_binary)

    # file_name = 'lstm_dropout0.5_units4.pth'
    # model_binary.save(file_name)

    model_binary = get_model_binary(0.25, 8)
    model_binary.fit(X_train, y_train_binary)

    file_name = 'lstm_dropout0.25_units8.pth'
    model_binary.save(file_name)

    # model_binary = get_model_binary(0.5, 8)
    # model_binary.fit(X_train, y_train_binary)

    # file_name = 'lstm_dropout0.5_units8.pth'
    # model_binary.save(file_name)

    model_binary.fit(X_train, y_train_binary)

    model_binary.save('lstm_baseline_model_binary_saved.pth')

    model_binary = load_model('lstm_dropout0.25_units8.pth')

    model_binary_test = load_model('/content/drive/MyDrive/CIS_530_PROJECT/lstm_dropout0.25_units8.pth')

    y_pred_binary_test = model_binary_test.predict(X_test)

    binary_pred_df_test = np.asarray((pd.DataFrame(y_pred_binary_test, columns=list_classes_binary) >= 0.5).astype(int))
    y_test_binary = df_test[list_classes_binary].values

    print("Precision: ", precision_score(y_test_binary, binary_pred_df_test))
    print("Recall: ", recall_score(y_test_binary, binary_pred_df_test))
    print("F1 Score: ", f1_score(y_test_binary, binary_pred_df_test))
    print("Accuracy Score: ", accuracy_score(y_test_binary, binary_pred_df_test))
    print(confusion_matrix(y_test_binary, binary_pred_df_test))
    
    y_pred_binary = model_binary.predict(X_test)

    threshold = 0.5
    binary_pred_df = (pd.DataFrame(y_pred_binary, columns=list_classes_binary) >= threshold).astype(int)
    binary_pred_df.to_csv('lstm_dropoutquarter_units8.csv')

    binary_np_pred = np.asarray(binary_pred_df)

    y_test_binary = df_test[list_classes_binary].values

    print("Precision: ", precision_score(binary_np_pred, y_test_binary))
    print("Recall: ", recall_score(binary_np_pred, y_test_binary))
    print("F1 Score: ", f1_score(binary_np_pred, y_test_binary))
    print("Accuracy Score: ", accuracy_score(binary_np_pred, y_test_binary))

    full_pred = pd.DataFrame(np.column_stack([df_test['comment_text'],binary_pred_df]), columns = ['comment_text', 'label'])

    full_pred.to_csv(test_label_output_path)
    