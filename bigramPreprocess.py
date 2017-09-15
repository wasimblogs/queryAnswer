import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import math
from nltk.util import ngrams

"""
Sentences with all stopwords will have [0 0 .. 0] BoW
"""
vocab = []  # global variable for vocabulary

# Stopword initialization
stop = stopwords.words('english')
stop.append("?")
stop.append("-")
stop.append("/")
stop.append(".")
stop.append(",")


def bigramVocabularyFromCorpus(inputString, debug=False, unigram=False):
    """
    Builds a vocabulary from coprus
    Takes string of corpus as input
    :return:
    """

    global vocab
    global inputSentence

    # Initialization of sentence tokenizer, word tokenizer, lemmatizer, stopword filter
    sentTokenizer = PunktSentenceTokenizer()
    wordTokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()

    # Check data type of input
    listDataType = []
    if type(inputString) == type(listDataType):
        inputSentence = inputString

    else:
        # lower string
        inputString = inputString.lower()

        # Input sentence preserves the original / unprocessed input of user
        inputSentence = sentTokenizer.tokenize(inputString)

    sentTokenizedFiltered = []
    vocabBigram = []
    for sentence in inputSentence:
        wordTokens = wordTokenizer.tokenize(sentence)
        wordTokens = [lemmatizer.lemmatize(word) for word in wordTokens if not word in stop]

        if not unigram:
            bigrams = list(ngrams(wordTokens, 2))

        for word in wordTokens:
            vocab.append(word)

        if not unigram:
            for word in bigrams:
                # print(word, "Length of bigram vocab : ", len(vocabBigram), " Length of bigram sentence : ", len(wordTokens))
                vocabBigram.append(word)
                wordTokens.append(word)

        # print("Bigram sentence: ", wordTokens)
        sentTokenizedFiltered.append(wordTokens)

    # Obtain set of words and sort it to obtain vocabulary
    vocab = set(vocab)
    vocab = sorted(vocab)
    if debug:
        print("Unigram vocab size : ", len(vocab))

    if not unigram:
        vocabBigram = set(vocabBigram)
        if debug:
            print("Bigram Vocab Size: ", len(vocabBigram))

        for word in vocabBigram:
            vocab.append(word)

        if debug:
            print("Total length of vocab: ", len(vocab))

    # vectorize(sentTokenizedFiltered, vocab)
    return sentTokenizedFiltered, vocab


def vectorize(sentTokens, vocab, debug=True):
    """
    # To compute the frequency occurence of words
    :param sentTokens: Array of tokenized sentences
    :param vocab: corpus vocabulary
    :return: count, presence
    """
    if debug:
        print("Vocab size : ", len(vocab))

    count = np.zeros((len(sentTokens), len(vocab)), np.float)
    # To count the number of documents a word is present in.
    presence = np.zeros((len(sentTokens), len(vocab)), np.float)

    for j in range(0, len(sentTokens)):
        # print(sentTokens[j])
        for word in sentTokens[j]:
            index = (vocab.index(word))
            count[j][index] += 1
            presence[j][index] = 1

    # Compute Term Frequency Inverse Document Frequency for given corpus
    # tfidf(count, presence, vocab, sentTokens)
    return count, presence


def tfidf(count, presence, vocab, sentTokens):
    """
    # Compute Term Frequency Inverse Document Frequency for given corpus

    :param count: Frequency count of words in corpus
    :param presence: Counts the number of documents in which a word is present
    :param vocab:
    :return: TF IDF of given corpus
    """

    presence = (presence.sum(axis=0))  # Column wise addition
    # The number of documents a term is present in.

    # Initialize paramters for TF IDF calculation
    global idfVec, tfidfVec
    tfidfVec = np.zeros_like(count)
    idfVec = np.zeros_like(presence)

    row, col = len(count), len(count[0])
    totalDocuments = row

    # Compute IDF Vector
    for j in range(0, col):
        # if presence[j]:
        idfVec[j] = (math.log(totalDocuments / presence[j]) + 1)
        # else:
        #     print("Null Presence", j)

    # Compute TF-IDF vector
    for i in range(0, row):
        for j in range(0, col):
            tfidfVec[i][j] = count[i][j] * idfVec[j]

        # Normalize vector
        tfidfVec[i] = normalizeVector(tfidfVec[i])

    # print("bag of words : " , tfidfVec)
    # analyze(tfidfVec, sentTokens, vocab)
    return tfidfVec


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


def analyze(tfidfVec, sentTokens, vocab, nTopWords=3, debug=True):
    """Finds the most important words in a sentence"""

    for vector, sentence in zip(tfidfVec, inputSentence):
        max = np.max(vector)
        sortedVector = sorted(vector)
        sortedVector.reverse()

        # Print n most important words in a sentence
        maxScores = (sortedVector[:nTopWords])
        result = []
        listVector = list(vector)
        for i in range(0, nTopWords):
            try:
                index = listVector.index(maxScores[i])
                result.append(vocab[index])
                listVector[index] = 0
            except:
                print("Bad Matches!")

        if debug:
            print("\n\n", result, "----", sentence)


def preProcessSent(sentence,unigram=False):
    """
    To lower. Stop word removal. Lemmatize. Return lemmatized input
    :param sentence: input which needs to be preprocessed
    :return: preprocessed sentence
    """

    # Lowercase
    sentence = sentence.lower()

    # word tokenizer
    wordTokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()

    wordTokens = wordTokenizer.tokenize(sentence)
    wordTokens = [lemmatizer.lemmatize(word) for word in wordTokens if word not in stop]

    if unigram:
        bigrams = list(ngrams(wordTokens, 2))

        for word in bigrams:
            wordTokens.append(word)

        # print("Bigram sentence: ", wordTokens)

    return wordTokens


def userInput():
    # User enters sentence
    # We calculate the tf idf of each sentence
    # and do consine similarity with each sentence in database.
    pass


def vectorizeSentence(wordTokens, topNMatch=10):
    print("Length of vocabulary : ", len(vocab))
    # print(idfVec)

    freq = np.zeros(len(vocab))
    for word in wordTokens:
        # If the vocabulary is not comprehensive user vocabulary may exceed database vocab
        try:
            index = (vocab.index(word))
            freq[index] += 1
        except:
            print("Vocabulary not present in corpus", word)

    for i in range(0, len(vocab)):
        freq[i] *= idfVec[i]

    # Normalize vector
    freq = normalizeVector(freq)

    scores = []
    for vector in tfidfVec:
        score = (vector * freq).sum(axis=0)
        scores.append(score)

    # Find the best N similar sentence / matches for this query
    for i in range(0, topNMatch):
        # try:
        max = np.max(scores)
        index = scores.index(max)
        scores[index] = 0
        print(i, "Best matches : ", max, inputSentence[index])
        # except:
        #     print("You broke the system. No matches found.")

    return freq
    # Reconstruct sentence to check if you counted right


def readCorpus(filename):
    corpus = []
    string = ""
    with open(filename, "r") as file:
        for line in file:
            # print("Line:",line)
            corpus.append(line)
            string += line
        print(len(corpus))

    file.close()
    return string, corpus


def buildQAModel(corpus, debug=False, unigram=False):
    """
    builds QA model from given corpus
    :param corpus:
    :return:
    """

    sentTokenizedFiltered, vocab = bigramVocabularyFromCorpus(corpus, debug=debug, unigram=True)
    count, presence = vectorize(sentTokenizedFiltered, vocab)
    tfidfVec = tfidf(count, presence, vocab, sentTokenizedFiltered)
    analyze(tfidfVec, sentTokenizedFiltered, vocab, 3, debug=debug)


def testAgainstQueries():
    from nltk.corpus import abc, genesis
    trainText = abc.raw("science.txt")
    trainText = genesis.raw("english-web.txt")

    string, input = readCorpus("faq.txt")
    buildQAModel(trainText)

    queries = ["How do I get ATM card?",
               "What is an ATM card?",

               "What is NRI ?",
               "What happens if I forget password?",
               "What are the difference between debit card and visa card?",
               "What is Bishal lost or forgotten?",
               "What is RFC account?",
               "How to reset account password?",
               "What are the benefits of using nsbl debit card or visa card?",
               "What happens when I lose the atm card?",
               "What are the daily withdrawal limits for the cards?",
               "To subscribe for NSBL DEBIT CARDS, our account holders can visit any of our branches and submit the duly filled card application form"]

    queries = ["Did god curse this earth?",
               "a man will leave his father and his mother",
               "Why did Jesus came to us?",
               "Who killed the egyptians?",
               "please forgive me joseph",
               "Where do I find salvation on this earth?",
               "What is God?"]

    for query in queries:
        processedQuery = preProcessSent(query)
        print("\n\n", "*" * 60, "\n\nQuestion : ", query, "\nMatches...")
        vectorizeSentence(processedQuery, 3)


if __name__ == "__main__":
    testAgainstQueries()
    string, input = readCorpus("faq.txt")
    buildQAModel(string, unigram=True)
