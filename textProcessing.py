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

# Initialization of sentence tokenizer, word tokenizer, lemmatizer, stopword filter
sentTokenizer = PunktSentenceTokenizer()
wordTokenizer = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Stopword initialization
stop = stopwords.words('english')
addStopwords = "? - / . \, , ' ’) ( ; the \" \, ,\" said \" \' r \“ “ ’ : ,” ') "" )^ [ ] ], ?\" )? } {  \) \( ):  = "" "" ?"" is the"
addStopwords = addStopwords.split()
stop = stop + addStopwords
stop = [stemmer.stem(word) for word in stop]

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
def vocabFromCorpus(inputString, showDebugInfo=False, computeBigram=False, minBigramFreq=1, minUnigramFreq=1):
    """
    Builds a vocabulary from coprus
    :param inputString: corpus as string or list of sentences
    Takes string of corpus as input
    :return: vocab
    """

    # Check data type of input
    listDataType = []
    if type(inputString) == type(listDataType):
        inputSentence = inputString

    else:
        # lower string
        inputString = inputString.lower()

        # Input sentence preserves the original / unprocessed input of user
        inputSentence = sentTokenizer.tokenize(inputString)

    vocab = []
    vocabBigram = []
    for index, sentence in enumerate(inputSentence):

        # Tokenize, stem and filter stopwords
        wordTokens = wordTokenizer.tokenize(sentence)
        wordTokens = [stemmer.stem(word.lower()) for word in wordTokens]
        wordTokens = [word for word in wordTokens if not word in stop]

        vocab = vocab + wordTokens

        if showDebugInfo:
            print("Processing {} / {} sentence : {}".format(index, len(inputSentence), sentence))
            print(wordTokens)

        if computeBigram:
            bigrams = list(ngrams(wordTokens, 2))
            # print(word, "Length of bigram vocab : ", len(vocabBigram), " Length of bigram sentence : ", len(wordTokens))
            vocabBigram = vocabBigram + bigrams

    # Obtain set of words and sort it to obtain vocabulary
    unigramCounter = Counter(vocab)
    usefulUnigramVocab = []
    for key, value in unigramCounter.items():
        if value > minUnigramFreq:
            # print(key, value)
            usefulUnigramVocab.append(key)

    vocab = sorted(set(vocab))

    if showDebugInfo:
        print("Unigram vocab size : ", len(vocab))
        print("Useful unigram vocab size : ", len(usefulUnigramVocab))

    # Append bigram vocabulary to unigram vocabulary
    usefulBigramVocab = []
    if computeBigram:
        bigramCounter = Counter(vocabBigram)

        for key, value in bigramCounter.items():
            if value > minBigramFreq:
                # print(key, value)
                usefulBigramVocab.append(key)

        vocabBigram = list(set(vocabBigram))
        # if showDebugInfo:
        if True:
            print("Bigram Vocab Size: ", len(vocabBigram))
            print("Useful Bigram Vocab Size: ", len(usefulBigramVocab))

        vocab = usefulUnigramVocab + usefulBigramVocab

        if showDebugInfo:
            print("Total length of vocab: ", len(vocab))

    return vocab


def tokenizeIntoWords(inputSentences, vocab, computeBigram=False, showDebugInfo=False,):
    """
    :param inputSentences: list of sentences from which stop words are to be removed
    :param type: list(str)
    :return: list of sentences tokenized into words. Words are stemmed and stopwords are filtered.
    """
    tokenizedOutput = []
    for sentence in inputSentences:
        words = wordTokenizer.tokenize(sentence)
        words = [stemmer.stem(word.lower()) for word in words]
        acceptedWords = [word for word in words if word in vocab]
        discardWords = [word for word in words if word in stop]

        if computeBigram:
            bigrams = list(ngrams(acceptedWords, 2))
            acceptedBigram = [word for word in bigrams if word in vocab]
            acceptedWords += acceptedBigram

        tokenizedOutput.append(acceptedWords)

        if showDebugInfo:
            print(sentence)
            print("Discarded : ", discardWords)
            print("Accepted : ", acceptedWords)
            print("\n\n")

    return tokenizedOutput


@timeit
def vectorize(sentTokens, vocab, showDebugInfo=False):
    """
    # To compute the frequency occurence of words and no of documents a certain word is present in.
    :param sentTokens: list of sentences tokenized into words
    :param type: list(list(str))
    :param vocab: corpus vocabulary
    :return: count, presence
    """
    if showDebugInfo:
        print("Vocab size : ", len(vocab))

    # Check if the data type is right
    dummyListType = []
    if type(sentTokens[0]) != type(dummyListType):
        print("Bad Data type sent. list(list(str)) expected!")
        return -1

    count = np.zeros((len(sentTokens), len(vocab)), np.float)
    # To count the number of documents a word is present in.
    presence = np.zeros((len(sentTokens), len(vocab)), np.float)

    for indexSent, tokenizedSent in enumerate(sentTokens):
        if showDebugInfo:
            print(tokenizedSent)

        for word in tokenizedSent:
            try:
                indexWord = (vocab.index(word))
                count[indexSent][indexWord] += 1
                presence[indexSent][indexWord] = 1
                # print("Indexed : ", word)
            except:
                # word not in vocab
                print(word, " not in vocab")
                pass

        if showDebugInfo:
            print("Len of tokens : {} \tSum of count: {} \n\n".format(len(tokenizedSent), np.sum(count[indexSent])))

    presence = (presence.sum(axis=0))  # Column wise addition

    # No word should have 0 presence because the vocab is built from corpus
    if showDebugInfo:
        for i, word in zip(presence, vocab):
            if not i:
                print("Presence : {} \tWord : {}".format(i, word))

    return count, presence


@timeit
def tfidf(count, presence, vocab):
    """
    # Compute Term Frequency Inverse Document Frequency for given corpus

    :param count: Frequency count of words in corpus
    :param presence: Counts the number of documents in which a word is present
    :return: TFIDF of given corpus
    """

    # Initialize paramters for TF IDF calculation
    tfidfVec = np.zeros_like(count)
    global idfVec
    idfVec = np.zeros_like(presence)

    # for present, word in zip(presence, vocab):
    #     print("Document : {} \tWord : {}".format(present, word))

    row, col = len(count), len(count[0])
    totalDocuments = row

    counts = 0
    # Compute IDF Vector
    for j in range(0, col):
        # There are no tokens with 0 presence
        try:
            idfVec[j] = (math.log(totalDocuments / presence[j]) + 1)
            # idfVec[j] = (math.log(totalDocuments / 100) + 1)
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


def genSynonyms(word, vocab):
    """
    :param word: word for which synonym is to be calculated
    :return: list of synonyms generated for the input word. Only returns the synonym if it is present in vocab
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


def nTopWordsinSentence(tfidfVec, sentTokens, vocab, n=3, showDebugInfo=False):
    """
    :param tfidfVec: 
    :param sentTokens: sentence tokenized into words -> list(words)
    :param vocab: vocab of corpus
    :param n: 
    :param showDebugInfo: 
    :return: list of n most important words in sorted order
    """

    sortedVector = sorted(tfidfVec)
    sortedVector.reverse()

    # Print n most important words in a sentence
    maxScores = (sortedVector[:n])
    nTopWordList = []
    listVector = list(tfidfVec)

    trueN = [word for word in sentTokens if word in vocab]
    trueN = len(trueN)
    if showDebugInfo:
        print("Only {} keywords".format(trueN))

    if trueN > 0:  # trueN can be 0 if no query words are present in vocab or are stopwords
        for i in range(0, n):
            try:
                index = listVector.index(maxScores[i])
                nTopWordList.append(vocab[index])
                listVector[index] = 0  # Two words might have same score. So make scores 0 as you read them
            except:
                print("Nan Found! Can't determine important words")
                break

            # For those sentences that do not have as many important words as user asked for
            if len(nTopWordList) == trueN:
                break
    else:
        return -1

    if showDebugInfo:
        print("\n\nNo of words in sentence : ", len(sentTokens), nTopWordList, "----", sentTokens)
        # print(tokenizedSent)

    return nTopWordList


def processQuery(sentence, vocab, computeBigram=False, showDebugInfo=False, insertSynonym=True):
    """
    To lower. Stop word removal. Stem. 
    :param sentence: input which needs to be preprocessed
    :param sentence: 
    :param vocab: 
    :param computeBigram: bool
    :param showDebugInfo: bool
    :param insertSynonym: generate synonym of words present in the sentence. Accepts bool
    :return: word tokens of sentence
    """
    # Lowercase
    sentence = sentence.lower()

    wordTokens = wordTokenizer.tokenize(sentence)
    wordTokens = [stemmer.stem(word) for word in wordTokens if word not in stop]
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
            syns = genSynonyms(word, vocab)
            synonyms = synonyms + syns

        # Do not repeat synonyms
        synonyms = list(set(synonyms))

        if showDebugInfo:
            print("Generated Synonyms : ", synonyms)

        # Query containing synonyms of terms present in query
        wordTokens = list(set(wordTokens + synonyms))

    return wordTokens


def vectorizeUserQuery(wordTokens, vocab, showDebugInfo=False):
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
            if showDebugInfo:
                print(word, "\t not present in vocab.")

    # Compute TF-IDF for input query
    for i in range(0, len(vocab)):
        freq[i] *= idfVec[i]

    # Normalize vector
    freq = normalizeVector(freq)
    return freq


def bestMatchForQuery(queryVec, tfidfVecQuestion, tfidfAnwer, topNMatch=10, considerAnswer=False):
    # Compute Similarity score of input query with all queries in database
    # inputQuestion contains unprocsesed input from corpus
    """
    :param queryVec: tfidfVec of query sentence
    :param tfidfVecQuestion: array of tfIdf of list of questions
    :param inputSentence: input sentence not tokenized
    :param topNMatch: 
    :return: queryMatchScore : list[score, index)
    """

    questionScores = []
    for vector in tfidfVecQuestion:
        score = (vector * queryVec).sum(axis=0)
        questionScores.append(score)

    # Find the best N similar sentence / matches for this query
    queryMatchScore = []
    for i in range(0, topNMatch):
        # try:
        maxScore = np.max(questionScores)
        index = questionScores.index(maxScore)

        if considerAnswer:
            ansScore = (tfidfAnwer[index] * queryVec).sum(axis=0)
            maxScore += ansScore

        queryMatchScore.append((maxScore, index))
        questionScores[index] = 0

    queryMatchScore.sort()
    queryMatchScore.reverse()

    # print("Top {} matches : ".format(topNMatch))
    # for score, index in queryMatchScore:
    #     print("Index : ", index, "score: %0.2f" % score, inputSentence[index])

    return queryMatchScore


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


def readBaseCorpus(filename=""):
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


def qaPipelineBaseCorpus(computeBigram=False, insertSynonym=False):
    """
    First finds set of similar questions and then evaluates their answer to find the best query match
    :return: best answer for the given query
    """

    # Base file which contains every information about a site except the QA pair
    inputBase = readBaseCorpus("ncell.txt")
    vocabBase = vocabFromCorpus(inputBase, computeBigram=computeBigram, showDebugInfo=False)

    # Answer
    stringAnswer, inputAnswer = readCorpus("ncellanswer.txt")
    vocabAnswer = vocabFromCorpus(inputAnswer, computeBigram=computeBigram, showDebugInfo=False)

    # Question
    vocabQuestion, inputQuestion = [], []
    stringQuestion, inputQuestion = readCorpus("ncellfaq.txt")
    inputQuestion = stringQuestion.split("\n")
    vocabQuestion = vocabFromCorpus(inputQuestion, computeBigram=computeBigram, showDebugInfo=False)

    # QA
    vocab = list(set(vocabQuestion + vocabBase + vocabAnswer))
    inputSentence = inputQuestion + inputBase + inputAnswer
    tokenizedSentence = tokenizeIntoWords(inputSentence,vocab)

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

    for query in queries:
        processedQuery = processQuery(query, vocab, computeBigram=computeBigram, insertSynonym=insertSynonym)
        queryVec = vectorizeUserQuery(processedQuery, vocab)
        print("\n\n", "*" * 60, "\n\nQuestion : ", query)
        importantWords = nTopWordsinSentence(queryVec, sentTokens=processedQuery, vocab=vocab, n=5)
        print("Keywords : ", importantWords)
        similarSents = bestMatchForQuery(queryVec, tfidfVecQuestion, tfidfVecAnswer, topNMatch=3)
        for score, sentId in similarSents:
            print("%0.2f" %score, inputQuestion[sentId])

if __name__ == "__main__":
    qaPipelineBaseCorpus(computeBigram=False, insertSynonym=True)
    pass
