from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as mplt
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

    return features, labels


if __name__ == '__main__':

    error = 0
    noOfWrongPreds = []
    dataPoints = []

    features, labels = preProcess()

    print(type(features)," ",type(labels))

    X_train, y_train = shuffle(features, labels, random_state=seed)

    clf = PassiveAggressiveClassifier(random_state=seed)

    clf.fit(X_train, y_train)
    predpa = clf.predict(X_train)

    for i in range(1, len(predpa), 1):
        if y_train[i] - predpa[i] != 0:
            error += 1

        if i % 50 == 0:
            noOfWrongPreds.append(error / (i+1))
            dataPoints.append(i+1)

    print("--------------------------PassiveAgressive--------------------------------")
    print(error)
    print(np.divide(error, len(y_train), dtype=np.float))
    plot(noOfWrongPreds, dataPoints, "distribution of wrong predictions passiveAgresssive")

    error = 0
    noOfWrongPreds = []
    dataPoints = []

    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    predsgdc = clf.predict(X_train)

    for i in range(1, len(predsgdc), 1):
        if y_train[i] - predsgdc[i] != 0:
            error += 1

        if i % 50 == 0:
            noOfWrongPreds.append(error / (i+1))
            dataPoints.append(i+1)

    print("--------------------------SGD--------------------------------")
    print(error)
    print(np.divide(error, len(y_train), dtype=np.float))
    plot(noOfWrongPreds, dataPoints, "distribution of wrong predictions SGD")







