from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import matplotlib as mplt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')


seed = 42
np.random.seed(seed)


def plot(noOfWrongPred, dataPoints, name):
    font_size = 14
    fig = plt.figure(dpi=100,figsize=(10, 6))
    mplt.rcParams.update({'font.size': font_size})
    plt.title("Distribution of wrong predictions", fontsize=font_size)
    plt.ylabel('Error rate', fontsize=font_size)
    plt.xlabel('Number of data points', fontsize=font_size)

    plt.plot(dataPoints, noOfWrongPred, label='Prediction', color='blue', linewidth=1.8)
    # plt.legend(loc='upper right', fontsize=14)

    plt.savefig(name+'.png')
    # plt.show()

def preProcess():

    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')

    features = vectorizer.fit_transform(newsgroups_data.data)
    labels= newsgroups_data.target

    print(newsgroups_data.target[:30])

    return features, labels


if __name__ == '__main__':

    features, labels = preProcess()

    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, shuffle=False, random_state=42)

    clf = PassiveAggressiveClassifier(random_state=seed)

    for i in range(len(train_y)):
        X, y = train_x[i:i + 1], train_y[i:i + 1]
        clf.partial_fit(X, y,classes=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]))


    print("--------------------------PassiveAgressive--------------------------------")
    predpa = clf.predict(test_x)
    print("Accuracy score")
    print(accuracy_score(test_y, predpa))
    print("F1 score")
    print(f1_score(test_y, predpa, average='macro'))
    print("Recall")
    print(recall_score(test_y, predpa, average='macro'))
    print("Precision")
    print(precision_score(test_y, predpa, average='macro'))



    clf = SGDClassifier()

    for i in range(len(train_y)):
        X, y = train_x[i:i + 1], train_y[i:i + 1]
        clf.partial_fit(X, y, classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))

    print("--------------------------SGD--------------------------------")
    predsgdc = clf.predict(test_x)
    print("Accuracy score")
    print(accuracy_score(test_y, predsgdc))
    print("F1 score")
    print(f1_score(test_y, predsgdc, average='macro'))
    print("Recall")
    print(recall_score(test_y, predsgdc, average='macro'))
    print("Precision")
    print(precision_score(test_y, predsgdc, average='macro'))





