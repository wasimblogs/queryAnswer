import nltk
import random
from nltk.corpus import movie_reviews
from collections import Counter
import numpy as np
import math
from bagofWords import timeit
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def normalizeVector(vector):
    """
    Returns normalized vector
    """
    # Checking for divide by zero error
    sum = (vector.sum(axis=0))
    if sum == 0:
        return vector

    else:
        rootOfSquareSum = np.sqrt((vector * vector).sum(axis=0))
        vector = vector / rootOfSquareSum
        return vector


@timeit
def vectorize(sentTokens, vocab, showDebugInfo=True):
    """
    # To compute the frequency occurence of words
    :param sentTokens: Array of tokenized sentences
    :param vocab: corpus vocabulary
    :return: count, presence
    """

    if showDebugInfo:
        print("Vocab size : ", len(vocab))
        print("Lenth of reviews : ", len(sentTokens))

    count = np.zeros((len(sentTokens), len(vocab)), np.float)
    # To count the number of documents a word is present in.
    presence = np.zeros((len(sentTokens), len(vocab)), np.float)
    print("Variables declared")

    noOfReviews = len(sentTokens)
    for j in range(0, noOfReviews):
        print("Freq Processing {} / {}".format(j, noOfReviews))
        for word in sentTokens[j]:
            index = (vocab.index(word))
            count[j][index] += 1
            presence[j][index] = 1
            # print(presence[j].sum(axis=0), count[j].sum(axis=0), len(sentTokens[j]))

    presence = (presence.sum(axis=0))  # Column wise addition
    # The number of documents a term is present in.

    if showDebugInfo:
        print("TF IDF calculation")

    # Initialize paramters for TF IDF calculation
    global idfVec, tfidfVec
    tfidfVec = np.zeros_like(count)
    idfVec = np.zeros_like(presence)

    row, col = len(count), len(count[0])
    totalDocuments = row

    if showDebugInfo:
        print("IDF calculation")

    # Compute IDF Vector
    for j in range(0, col):
        idfVec[j] = (math.log(totalDocuments / presence[j]) + 1)
        # print("IDF Processing {} / {}".format(j, (col)))
        # print(idfVec[j], totalDocuments, presence[j])

    if showDebugInfo:
        print("TF IDF multiplication")

    # Compute TF-IDF vector
    for i in range(0, row):
        print("TF IDF Processing {} / {}".format(i, noOfReviews))
        for j in range(0, col):
            tfidfVec[i][j] = count[i][j] * idfVec[j]

        # Normalize vector
        tfidfVec[i] = normalizeVector(tfidfVec[i])

    return tfidfVec


@timeit
def collectData(nTopReviews=1000):
    # Initialize stemmer and lemmatizer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Read all the movie review data from two folders pos and neg
    category = (movie_reviews.categories())
    documents = []
    for category in movie_reviews.categories():
        for file in movie_reviews.fileids(category):
            review = movie_reviews.words(file)
            reviewProcessed = [stemmer.stem(word) for word in review]
            documents.append((reviewProcessed, category))

    # Shuffle movie review order so that pos and neg reviews are interspersed
    random.shuffle(documents)

    listOfReviews = []
    vocab = []
    for review, rating in documents[:nTopReviews]:
        listOfReviews.append(review)
        vocab = vocab + list(set(review))

    # Sorting and set operation to obtain vocab of corpus
    vocab = sorted(list(set(vocab)))
    print(len(vocab))

    # Obtaining TF IDF of movie reviews
    print("Now on to vectorizing")
    tfidfVec = vectorize(listOfReviews[:nTopReviews], vocab, showDebugInfo=True)

    # Getting feature, labels to train and test
    featureSets = []
    for i in range(0, nTopReviews):
        vec = list(tfidfVec[i])

        feature = {}
        for word, score in zip(vocab, vec):
            feature[word] = score

        rating = documents[i][1]
        featureSets.append((feature, rating))

    # Dividing data into trainSet and testSet
    trainingSet = featureSets[int(0.5 * nTopReviews):]
    testSet = featureSets[:int(0.5 * nTopReviews)]

    # Traingin classifer and testing accuracy
    classifier = nltk.NaiveBayesClassifier.train(trainingSet)
    accuracy = nltk.classify.accuracy(classifier, testSet) * 100

    print("Accuracy : {} %".format(accuracy))


if __name__ == '__main__':
    collectData()
