import nltk
import random
from nltk.corpus import movie_reviews
from collections import Counter
import numpy as np
import math
from bagofWords import timeit
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import bigrams
from nltk.tokenize import WordPunctTokenizer


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
def collectData(nTopReviews=10, computeBigram=False, lowFrequencyThresh=1):
    """
    Collect data from corpus. Form vocab. Stemming and stopword removal
    :return : listOfText, vocab
    :param lowFrequencyThresh: remove words with frequency equal or below this value
    """

    # stopword collection
    stopwordString = "have films was ; -- movie or this in - , . '  : of and a the to is s it that ( ) as film with for hi thi i he but be on movi are t by one an who habe you from at wa they ha her all charact there ? so out about up what"
    pronoun = "all another any anybody anyone anything both each either everybody everyone everything few he her hers herself him himself his I it its itself me my myself other our ours ourselves she somebody someone something that their theirs them themselves these they this those us we what whatever which whichever who whoever whom whomever whose you your yours yourself yourselves"

    pronoun = pronoun.split()
    stopword = stopwordString.split(" ")
    stopword.append('"')
    stopword = stopword + pronoun

    # Initialize stemmer and lemmatizer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    wordTokenizer = WordPunctTokenizer()

    # # Read all the movie review data from two folders pos and neg
    global documents
    documents = []
    # category = (movie_reviews.categories())
    # for category in movie_reviews.categories():
    #     for file in movie_reviews.fileids(category):
    #         review = movie_reviews.words(file)
    #         reviewProcessed = [stemmer.stem(word) for word in review]
    #         documents.append((reviewProcessed, category))

    # Read all the reviews from positive.txt and negative.txt
    short_pos = open("Dataset/positive.txt", "r").read()
    short_neg = open("Dataset/negative.txt", "r").read()

    for r in short_pos.split('\n'):
        wordTokens = wordTokenizer.tokenize(r)
        documents.append((wordTokens, "pos"))

    for r in short_neg.split('\n'):
        wordTokens = wordTokenizer.tokenize(r)
        documents.append((wordTokens, "neg"))

    print("Read reviews from database")

    # Shuffle movie review order so that pos and neg reviews are interspersed
    random.shuffle(documents)

    if nTopReviews > len(documents) or nTopReviews == 0:
        nTopReviews = len(documents)

    listOfReviews = []
    vocab = []
    for review, rating in documents[:nTopReviews]:
        try:
            listOfReviews.append(review)
            vocab = vocab + list(review)
        except:
            print("Probably I devoured all the reviews")

    # Sorting and set operation to obtain vocab of corpus
    print("Unprocessed Corpus Size : ", len(vocab))

    # Count word frequency to remove rare words from vocab and corpus
    counts = (Counter(vocab))

    vocab = list(set(vocab))
    print("Vocabulary size : ", len(vocab))

    # Add one time occuring word to stopword list
    oneTimeWords = [key for key in counts if counts[key] <= lowFrequencyThresh]
    stopword = stopword + oneTimeWords

    # Filter stopwords and rare words from vocab
    vocab = [word for word in vocab if word not in stopword]
    print("Vocabulary size after removing rare words: ", len(vocab))

    # Filter stopwords and rare words from text / reviews
    listOfReviewsStopworFiltered = []
    totalNoWords = 0
    vocabBigram = []
    for sent in listOfReviews:
        sentStopwordFiltered = [word for word in sent if word not in stopword]

        if computeBigram:
            bigram = list(bigrams(sentStopwordFiltered))
            sentStopwordFiltered = sentStopwordFiltered + bigram
            vocabBigram = vocabBigram + bigram

        listOfReviewsStopworFiltered.append(sentStopwordFiltered)
        totalNoWords = totalNoWords + len(sentStopwordFiltered)

    if computeBigram:
        vocab = vocab + list(set(vocabBigram))
        print("Bigram Vocab Size : ", len(vocab))

    listOfReviews = listOfReviewsStopworFiltered
    print("New Corpus Size : ", totalNoWords)
    return listOfReviews, vocab


def trainAndTest(listOfReviews, vocab, kFold=5):
    """
    :param listOfReviews: list of text / emails / sentence / moview reviews
    :param vocab: vocab of corpus
    :param kFold: k for cross validation
    :return:
    """

    # Obtaining TF IDF of movie reviews
    # print("Now on to vectorizing")
    # tfidfVec = vectorize(listOfReviews, vocab, showDebugInfo=True)

    # Getting feature, labels to train and test
    featureSets = []
    noOfReviews = len(listOfReviews)
    for i in range(0, noOfReviews):
        # vec = list(tfidfVec[i])

        feature = {}
        # for word, score in zip(vocab, vec):
        #     # feature[word] = score
        #     feature[word] = word in listOfReviews[i]

        for word in vocab:
            feature[word] = word in listOfReviews[i]

        rating = documents[i][1]
        featureSets.append((feature, rating))


    # k fold cross validation
    accuracySum = 0
    for i in range(0, kFold):
        random.shuffle(featureSets)
        # Dividing data into trainSet and testSet
        trainingSet = featureSets[int(0.5 * noOfReviews):]
        testSet = featureSets[:int(0.5 * noOfReviews)]

        # Traingin classifer and testing accuracy
        classifier = nltk.NaiveBayesClassifier.train(trainingSet)

        # To test if desired keywords caused the result or just random guesses
        # (classifier.show_most_informative_features(100))
        accuracy = nltk.classify.accuracy(classifier, testSet) * 100
        print("Accuracy : {} %".format(accuracy))

        accuracySum = accuracySum + accuracy

    print("Average Accuracy : ", (accuracySum)/kFold)
    return classifier


def preprocessUserInput(inputString):
    """
    Test movie reviews from live users in production phase
    :param inputString: text / movie review / email
    :return: input ready for classifer
    """
    pass


def testClassifer(inputQuery):
    """
    Testing classifier
    :param inputQuery:
    :return:
    """
    global classifier
    pass


if __name__ == '__main__':
    listOfReviews, vocab = collectData(1000, computeBigram=True, lowFrequencyThresh=1)
    classifier = trainAndTest(listOfReviews, vocab,kFold=10)
