import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import math
from nltk.util import ngrams
import time  # used to time profile methods
from nltk.corpus import wordnet
from collections import Counter

"""
Objective : To find the best match for a given question.

Things noticed and things to treat with caution
- Sentences with all stopwords will have [0 0 .. 0] BoW
- Make stopword list comprehensive
- Deal with spell errors
- Deal with user queries containing synonyms
- Use the best lemmatizer / stemmer
- Use consistent preprocessing for everything
- Use of global variables sparingly
"""

# Stopword initialization
stop = stopwords.words('english')
addStopwords = "? - / . \, , ' ’) ( ; the \" \, ,\" said \" \' r \“ “ ’ : ,” ') "
addStopwords = addStopwords.split(" ")
stop = stop + addStopwords
stop.append('the')

# Initialization of sentence tokenizer, word tokenizer, lemmatizer, stopword filter
sentTokenizer = PunktSentenceTokenizer()
wordTokenizer = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def timeit(method):
    """
    A decorator used for time profiling functions and methods
    :param method:
    :return: time in ms for a method
    """

    def timed(*args, **kwargs):
        timeStart = time.time()
        result = method(*args, **kwargs)
        timeEnd = time.time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((timeEnd - timeStart) * 1000)
        else:
            print('%r %2.2f ms' % (method.__name__, (timeEnd - timeStart) * 1000))
        return result

    return timed


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
def vocabularyFromCorpus(inputString, showDebugInfo=False, computeBigram=False):
    """
    Builds a vocabulary from coprus
    Takes string of corpus as input
    :return: vocab
    """

    vocab = []

    # Check data type of input
    listDataType = []
    if type(inputString) == type(listDataType):
        inputSentence = inputString

    else:
        # lower string
        inputString = inputString.lower()

        # Input sentence preserves the original / unprocessed input of user
        inputSentence = sentTokenizer.tokenize(inputString)

    vocabBigram = []
    for sentence in inputSentence:
        wordTokens = wordTokenizer.tokenize(sentence)
        wordTokens = [lemmatizer.lemmatize(word.lower()) for word in wordTokens if not word in stop]

        vocab = vocab + wordTokens

        if computeBigram:
            bigrams = list(ngrams(wordTokens, 2))
            # print(word, "Length of bigram vocab : ", len(vocabBigram), " Length of bigram sentence : ", len(wordTokens))
            vocabBigram = vocabBigram + bigrams
            wordTokens = wordTokens + bigrams

    # Obtain set of words and sort it to obtain vocabulary
    vocab = sorted(set(vocab))

    if showDebugInfo:
        print("Unigram vocab size : ", len(vocab))

    # Append bigram vocabulary to unigram vocabulary
    if computeBigram:
        vocabBigram = list(set(vocabBigram))
        if showDebugInfo:
            print("Bigram Vocab Size: ", len(vocabBigram))

        vocab = vocab + vocabBigram

        if showDebugInfo:
            print("Total length of vocab: ", len(vocab))

    return vocab


def tokenizeIntoWords(inputSentences):
    """
    :param inputSentences: list of sentences from which stop words are to be removed
    :return: list of processed sentences 
    """
    tokenizedOutput = []
    for sentence in inputSentences:
        words = wordTokenizer.tokenize(sentence)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in stop]
        tokenizedOutput.append(words)
        # print(words)

    return tokenizedOutput


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

    count = np.zeros((len(sentTokens), len(vocab)), np.float)
    # To count the number of documents a word is present in.
    presence = np.zeros((len(sentTokens), len(vocab)), np.float)

    for j in range(0, len(sentTokens)):
        for word in sentTokens[j]:
            try:
                index = (vocab.index(word))
                count[j][index] += 1
                presence[j][index] = 1
            except:
                # word not in vocab
                print(word, " not in vocab")
                pass

                # print("*" * 30, "\n")
                # print(np.sum(presence[j]), len(sentTokens[j]), sentTokens[j])

    presence = (presence.sum(axis=0))  # Column wise addition

    # No word should have 0 presence because the vocab is built from corpus
    for i, word in zip(presence, vocab):
        if not i:
            print(i, word)

    # The number of documents a term is present in.

    return count, presence


@timeit
def tfidf(count, presence, vocab):
    """
    # Compute Term Frequency Inverse Document Frequency for given corpus

    :param count: Frequency count of words in corpus
    :param presence: Counts the number of documents in which a word is present
    :return: TF IDF of given corpus
    """

    # Initialize paramters for TF IDF calculation
    tfidfVec = np.zeros_like(count)
    global idfVec
    idfVec = np.zeros_like(presence)

    row, col = len(count), len(count[0])
    totalDocuments = row

    counts = 0
    # Compute IDF Vector
    for j in range(0, col):
        # There are no tokens with 0 presence
        try:
            idfVec[j] = (math.log(totalDocuments / presence[j]) + 1)
        except:
            print("Divide by zero encountered ", vocab[j])
            print(counts, "Null presence", j, vocab[j])
            counts += 1

    # Compute TF-IDF vector
    for i in range(0, row):
        for j in range(0, col):
            tfidfVec[i][j] = count[i][j] * idfVec[j]

        # Normalize vector
        tfidfVec[i] = normalizeVector(tfidfVec[i])

    return tfidfVec


def calcSynonyms(word, vocab):
    """
    :param word: word for which synonym is to be calculated
    :return: list of synonym for the input word
    """

    synonyms = []

    # Compute synonym
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            word = l.name()
            synonyms.append(word)

    # Consider only those synonyms that are present in vocabulary of the corpus
    synonyms = [word for word in synonyms if word in vocab]
    synonyms = list(set(synonyms))

    return synonyms


def findTopWords(tfidfVec, sentTokens, inputSentence, vocab, nTopWords=3, showDebugInfo=True):
    """
    Finds the most important words in a sentence
    This module will phase out slowly in favor of nTopWordsinSentence
    """

    print("No of sentences : ", len(inputSentence))
    count = 0

    for vector, sentence, tokenizedSent in zip(tfidfVec, inputSentence, sentTokens):
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
                listVector[index] = 0  # Two words might have same score. So make scores 0 as you read them
            except:
                print("Nan Found! Can't determine important words")
                break

        if showDebugInfo:
            print("\n\n", count, "/", len(inputSentence), result, "----", sentence)
            # print(tokenizedSent)

        count += 1


def nTopWordsinSentence(tfidfVec, sentTokens, vocab, nTopWords=3, showDebugInfo=False):
    """Returns n most important words in a sentence"""

    sortedVector = sorted(tfidfVec)
    sortedVector.reverse()

    # Print n most important words in a sentence
    maxScores = (sortedVector[:nTopWords])
    nTopWordList = []
    listVector = list(tfidfVec)

    trueNLimit = [word for word in sentTokens if word in vocab]

    for i in range(0, nTopWords):
        try:
            index = listVector.index(maxScores[i])
            nTopWordList.append(vocab[index])
            listVector[index] = 0  # Two words might have same score. So make scores 0 as you read them
        except:
            print("Nan Found! Can't determine important words")
            break

        # For those sentences that do not have as many important words as user asked for
        if len(nTopWordList) == len(trueNLimit):
            break

    if showDebugInfo:
        print("\n\n", len(sentTokens), nTopWordList , "----", sentTokens)
        # print(tokenizedSent)

    return nTopWordList


def processQuery(sentence, vocab, computeBigram=False, showDebugInfo=False, insertSynonym=True):
    """
    To lower. Stop word removal. Lemmatize. Return lemmatized input
    :param sentence: input which needs to be preprocessed
    :return: word tokens of sentence
    """

    # Lowercase
    sentence = sentence.lower()

    wordTokens = wordTokenizer.tokenize(sentence)
    wordTokens = [lemmatizer.lemmatize(word) for word in wordTokens if word not in stop]
    listOfWords = set(wordTokens)  # used to calculate synonyms

    if computeBigram:
        bigrams = list(ngrams(wordTokens, 2))
        wordTokens = wordTokens + bigrams

        if showDebugInfo:
            print("Bigram sentence: ", wordTokens)

    # If insert synonyms of words present in the sentence
    if insertSynonym:
        synonyms = []
        for word in listOfWords:
            syns = calcSynonyms(word, vocab)
            synonyms = synonyms + syns

        # Do not repeat synonyms
        synonyms = list(set(synonyms))

        # Query containing synonyms of terms present in query
        wordTokens = list(set(wordTokens + synonyms))

    if showDebugInfo:
        print(wordTokens)

    return wordTokens


def userInput():
    # User enters sentence
    # We calculate the tf idf of each sentence
    # and do consine similarity with each sentence in database.
    pass


def vectorizeUserQuery(wordTokens, vocab):
    """
    Vectorize user query and compute similarity with database questions
    :param wordTokens: tokenized user query
    :param topNMatch: how many best matches to show
    :return: TF-IDF of user query
    """
    # print("Length of vocabulary : ", len(vocab))

    freq = np.zeros(len(vocab))
    for word in wordTokens:
        # If the vocabulary is not comprehensive user vocabulary may exceed database vocab
        try:
            index = (vocab.index(word))
            freq[index] += 1
        except:
            print("Vocabulary not present in corpus : ", word)

    # Compute TF-IDF for input query
    for i in range(0, len(vocab)):
        freq[i] *= idfVec[i]

    # Normalize vector
    freq = normalizeVector(freq)
    return freq


def findSimilarSentence(queryVec, tfidfVecQuestion, tfidfAnwer, inputSentence, topNMatch=10):
    # Compute Similarity score of input query with all queries in database
    # inputQuestion contains unprocsesed input from corpus

    questionScores = []
    for vector in tfidfVecQuestion:
        score = (vector * queryVec).sum(axis=0)
        questionScores.append(score)

    # Find the best N similar sentence / matches for this query
    bestIndices = []
    for i in range(0, topNMatch):
        # try:
        maxScore = np.max(questionScores)
        index = questionScores.index(maxScore)
        bestIndices.append((maxScore, index))
        questionScores[index] = 0

    finalScore = []
    for score, index in bestIndices:
        answerScore = (tfidfAnwer[index] * queryVec).sum(axis=0)
        finalScore.append((score + answerScore, index))

    finalScore.sort()
    finalScore.reverse()

    print("Top {} matches : ".format(topNMatch))
    for score, index in finalScore:
        print("Index : ", index, "score: %0.2f" % score, inputSentence[index])


@timeit
def readCorpus(filename):
    """
    :param filename: a question is written as a line in file. Reads that file
    :return: returns list of questions in string and list format
    """
    list = []
    string = ""
    with open(filename, "r") as file:
        for line in file:
            # print("Line:",line)
            string += line
        print("No. of questions:", len(list))

    file.close()

    list = string.split("###")
    return string, list


@timeit
def readAnswers(filename, showDebugInfo=False):
    """
    :param filename: a question is written as a line in file. Reads that file
    :return: returns list of questions
    """
    corpus = []
    string = ""
    with open(filename, "r") as file:
        for line in file:
            # print("Line:",line)
            corpus.append(line)
            string += line

    file.close()

    answers = string.split("###")

    if showDebugInfo:
        print("No. of answers", len(answers))
        for answer in answers:
            print(answers.index(answer), answer)
            print("\n")

    return answers


@timeit
def buildQAModel(corpus, showDebugInfo=False, computeBigram=False):
    """
    builds QA model from given corpus
    :param corpus:
    :return:
    """

    print("1/4 - Computing vocabulary from corpus")
    vocab = vocabularyFromCorpus(corpus, showDebugInfo=showDebugInfo,
                                 computeBigram=computeBigram)

    # Needs work here
    sentTokenizedFiltered = tokenizeIntoWords(corpus)

    print("2/4 - Vectorizing queries")
    count, presence = vectorize(sentTokenizedFiltered, vocab)

    print("3/4 - Bag of words with weight of words computed using TD-IDF")
    tfidfVec = tfidf(count, presence, vocab, sentTokenizedFiltered)

    print("4/4 - Finding most important words in queries")
    findTopWords(tfidfVec, sentTokenizedFiltered, vocab, 3, showDebugInfo=True)


def qaPipeline(computeBigram=False, insertSynonym=False):
    """
    First finds set of similar questions and then evaluates their answer to find the best query match
    :return: best answer for the given query
    """

    # Answer
    stringAnswer, inputAnswer = readCorpus("ncellanswer.txt")
    vocabAnswer = vocabularyFromCorpus(inputAnswer, computeBigram=computeBigram, showDebugInfo=True)

    # Question
    vocabQuestion, inputQuestion = [], []
    stringQuestion, inputQuestion = readCorpus("ncellfaq.txt")
    inputQuestion = stringQuestion.split("\n")
    vocabQuestion = vocabularyFromCorpus(inputQuestion, computeBigram=computeBigram, showDebugInfo=True)

    # QA
    vocab = list(set(vocabQuestion + vocabAnswer))
    inputSentence = inputQuestion + inputAnswer
    tokenizedSentence = tokenizeIntoWords(inputSentence)

    # Vectorize or Document to matrix
    count, presence = vectorize(tokenizedSentence, vocab)

    # TF IDF
    tfidfVec = tfidf(count, presence, vocab)
    tfidfVecQuestion = tfidfVec[:len(inputQuestion)]
    tfidfVecAnswer = tfidfVec[len(inputQuestion):]

    # # Checking if sentences are tokenized properly
    # for tSent, sent in zip(tokenizedSentence, inputSentence):
    #     print("->> ", tSent)
    #     print("->>", sent)
    #     print("\n\n", "*" * 40)

    # Make sure you're sending tokenized and unprocessed sentence of same sentence
    # findTopWords(tfidfVec=tfidfVec, sentTokens=tokenizedSentence, inputSentence=inputSentence, vocab=vocab)

    queries = [
        "How to activate 4G?",
        "What are cities with 4G network coverage?",
        "Do you sell phones?",
        "Where can I buy ringtone?",
        "What happens if I lose my SIM card?",
        "What is the difference between 4G and 3G sim card?",
        "What is the difference between tariffs of 3G and 4G sim card?",
        "I get busy all the time ringtone?"]

    for query in queries:
        processedQuery = processQuery(query, vocab, computeBigram=computeBigram, insertSynonym=insertSynonym)
        queryVec = vectorizeUserQuery(processedQuery, vocab)
        print("\n\n", "*" * 60, "\n\nQuestion : ", query)
        findSimilarSentence(queryVec, tfidfVecQuestion, tfidfVecAnswer, inputSentence=inputSentence, topNMatch=3)


def qaPipelineBaseCorpus(computeBigram=False, insertSynonym=False):
    """
    First finds set of similar questions and then evaluates their answer to find the best query match
    :return: best answer for the given query
    """

    # Base file which contains every information about a site except the QA pair
    inputBase = readBaseCorpus("ncell.txt")
    vocabBase = vocabularyFromCorpus(inputBase, computeBigram=computeBigram, showDebugInfo=True)

    # Answer
    stringAnswer, inputAnswer = readCorpus("ncellanswer.txt")
    vocabAnswer = vocabularyFromCorpus(inputAnswer, computeBigram=computeBigram, showDebugInfo=True)

    # Question
    vocabQuestion, inputQuestion = [], []
    stringQuestion, inputQuestion = readCorpus("ncellfaq.txt")
    inputQuestion = stringQuestion.split("\n")
    vocabQuestion = vocabularyFromCorpus(inputQuestion, computeBigram=computeBigram, showDebugInfo=True)

    # QA
    vocab = list(set(vocabQuestion + vocabBase + vocabAnswer))
    inputSentence = inputQuestion + inputBase + inputAnswer
    tokenizedSentence = tokenizeIntoWords(inputSentence)

    # Vectorize or Document to matrix
    count, presence = vectorize(tokenizedSentence, vocab)

    # TF IDF
    tfidfVec = tfidf(count, presence, vocab)
    tfidfVecQuestion = tfidfVec[:len(inputQuestion)]
    tfidfVecAnswer = tfidfVec[-1 * len(inputQuestion):]

    # # Checking if sentences are tokenized properly
    # for tSent, sent in zip(tokenizedSentence, inputSentence):
    #     print("->> ", tSent)
    #     print("->>", sent)
    #     print("\n\n", "*" * 40)

    # # Make sure you're sending tokenized and unprocessed sentence of same sentence
    # findTopWords(tfidfVec=tfidfVec, sentTokens=tokenizedSentence, inputSentence=inputSentence, vocab=vocab)

    queries = [
        "How to activate 4G?",
        "What are cities with 4G network coverage?",
        "Do you sell phones?",
        "Where can I buy ringtone?",
        "What happens if I lose my SIM card?",
        "What is the difference between 4G and 3G sim card?",
        "What is the difference between tariffs of 3G and 4G sim card?",
        "I get busy all the time ringtone?"]

    inputSentence = inputQuestion + inputAnswer
    for query in queries:
        processedQuery = processQuery(query, vocab, computeBigram=computeBigram, insertSynonym=insertSynonym)
        queryVec = vectorizeUserQuery(processedQuery, vocab)
        print("\n\n", "*" * 60, "\n\nQuestion : ", query)
        importantWords = nTopWordsinSentence(queryVec, sentTokens=processedQuery, vocab=vocab, nTopWords=5)
        print("Keywords : ", importantWords)
        findSimilarSentence(queryVec, tfidfVecQuestion, tfidfVecAnswer, inputSentence=inputQuestion, topNMatch=3)


def readBaseCorpus(filename="ncell.txt"):
    """
    :param filename: filename containing base corpus. Base corpus is used to normalize word importance. It contains
    all the text in a website. 
    :return: list of sentences
    """
    with open(filename) as file:
        string = ""
        for line in file:
            if len(line) > 20:  # store the sentence only if its' len is greater than 20
                string += line

    # Tokenize string into sentence
    sentences = sentTokenizer.tokenize(string)

    # for sentence in sentences:
    #     print("*",sentence)

    return sentences


if __name__ == "__main__":
    # qaPipeline()
    # qaPipeline(computeBigram=False, insertSynonym=True)
    qaPipelineBaseCorpus(computeBigram=False, insertSynonym=True)
    pass
