import os
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def pre_process():
    direc = "enron/emails/"
    files = os.listdir(direc)
    emails = [direc+email for email in files]

    words = []
    temp_email_text = []
    labels = []

    for email in emails:
        if "ham" in email:
            labels.append(0)
        else:
            labels.append(1)
        f = open(email,encoding="utf8", errors='ignore')
        blob = f.read()
        all_text = ''.join([text for text in blob if text not in punctuation])
        all_text = all_text.split('\n')
        all_text = ''.join(all_text)
        temp_text = all_text.split(" ")

        for word in temp_text:
            if word.isalpha():
                temp_text[temp_text.index(word)] = word.lower()

        temp_text = list(filter(None, temp_text))
        temp_text = ' '.join([i for i in temp_text if not i.isdigit()])
        words += temp_text.split(" ")
        temp_email_text.append(temp_text)

    dictionary = Counter(words)

    #deleting spaces
    del dictionary[""]
    sorted_split_words = sorted(dictionary, key=dictionary.get, reverse=True)
    vocab_to_int = {c: i for i, c in enumerate(sorted_split_words, 1)}

    message_ints = []
    for message in temp_email_text:
        temp_message = message.split(" ")
        message_ints.append([vocab_to_int[i] for i in temp_message])

    seq_length = 3425
    num_messages = len(temp_email_text)
    features = np.zeros([num_messages,seq_length], dtype=int)
    for i, row in enumerate(message_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features, np.array(labels)



def train_test():

    features, labels = pre_process()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False, random_state=42)

    print(X_train.shape, "   ", X_test.shape)
    print(y_train.shape, "   ", y_test.shape)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    p = clf.predict(X_test)

    print("----------------------Logistic Regression------------------------------")
    print("Accuracy score")
    print(accuracy_score(y_test, p))
    print("F1 score")
    print(f1_score(y_test, p, average='macro'))
    print("Recall")
    print(recall_score(y_test, p, average='macro'))
    print("Precision")
    print(precision_score(y_test, p, average='macro'))

    mlnb = MultinomialNB()
    mlnb.fit(X_train, y_train)
    pred = mlnb.predict(X_test)

    print("----------------------MultinomialNB------------------------------")
    print("Accuracy score")
    print(accuracy_score(y_test, pred))
    print("F1 score")
    print(f1_score(y_test, pred, average='macro'))
    print("Recall")
    print(recall_score(y_test, pred, average='macro'))
    print("Precision")
    print(precision_score(y_test, pred, average='macro'))


if __name__ == '__main__':
    train_test()

#----------------------Logistic Regression------------------------------
# Accuracy score
# 0.5478260869565217
# F1 score
# 0.53451854699212
# Recall
# 0.5754210644509319
# Precision
# 0.5630349099099099
# ----------------------MultinomialNB------------------------------
# Accuracy score
# 0.4821256038647343
# F1 score
# 0.4734304285757677
# Recall
# 0.5164383561643835
# Precision
# 0.5139969252655974






