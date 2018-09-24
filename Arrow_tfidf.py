import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as mplt
mplt.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

seed = 42
np.random.seed(seed)


class AROW:

    def __init__(self, nb_class, d):
        self.w = np.zeros((nb_class, d))
        self.sigma = np.identity(d)
        self.r = 1
        self.nb_class = nb_class

    def fit(self, X, y):
        w = np.copy(self.w)
        sigma = np.copy(self.sigma)

        # y = ((y - y.min()) * (1/(y.max() - y.min()) * (nb_class-1))).astype('uint8')

        F_t = np.dot(self.w, X.T)

        # compute hinge loss and support vector
        F_s = np.copy(F_t)
        F_s[y] = -np.inf
        s_t = np.argmax(F_s)
        m_t = F_t[y] - F_t[s_t]
        v_t = np.dot(X, np.dot(sigma, X.T))
        l_t = np.maximum(0, 1 - m_t)  # hinge loss

        # update weights
        if l_t > 0:
            beta_t = 1 / (v_t + self.r)
            alpha_t = l_t * beta_t
            self.w[y] = w[y] + (alpha_t * np.dot(sigma, X.T).T)
            self.w[s_t] = w[s_t] - (alpha_t * np.dot(sigma, X.T).T)
            self.sigma = sigma - beta_t * np.dot(np.dot(sigma, X.T), np.dot(X, sigma))

    def predict(self, X):
        return np.argmax(np.dot(self.w, X.T), axis=0)

def preProcess():

    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    feature_set= vectorizer.fit_transform(newsgroups_data.data)

    labels = newsgroups_data.target

    return feature_set, labels


def plot(noOfWrongPred, dataPoints, name):
    font_size = 14
    fig = plt.figure(dpi=100,figsize=(10, 6))
    mplt.rcParams.update({'font.size': font_size})
    plt.title("Distribution of wrong predictions", fontsize=font_size)
    plt.ylabel('Number of wrong predictions', fontsize=font_size)
    plt.xlabel('Number of data points', fontsize=font_size)

    plt.plot(dataPoints, noOfWrongPred, label='Prediction', color='blue', linewidth=1.8)
    plt.legend(loc='upper right', fontsize=14)

    plt.savefig(name+'.png')


if __name__ == '__main__':

    features, labels = preProcess()

    X_train, y_train = shuffle(features, labels, random_state=seed)

    n, d = X_train.shape
    nb_class = len(set(labels))

    arow = AROW(nb_class, d)

    error = 0
    noOfWrongPreds = []
    dataPoints = []
    for i in range(n):
        X, y = X_train[i:i + 1], y_train[i:i + 1]

        p_y = arow.predict(X)
        arow.fit(X, y)

        if y-p_y != 0:
            error += 1

        if i % 50 == 0:
            print(error)
            print(i)
            print(i+1)
            noOfWrongPreds.append(error / (i+1))
            dataPoints.append(i+1)

    print(error)
    print(np.divide(error, n, dtype=np.float))
    plot(noOfWrongPreds, dataPoints, "distribution of wrong predictions Arrow-tfidf")

