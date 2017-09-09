import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import math



def corpusToVocabulary(inputString):
    """
    Builds a vocabulary from coprus
    Takes string of corpus as input
    Converts into word token
    :return:
    """

    # lower string
    inputString = inputString.lower()

    # Tokenize string into sentence
    sentTokenizer = PunktSentenceTokenizer()
    sentTokenized = sentTokenizer.tokenize(inputString)
    # print("Sentence Tokenized : ", sentTokenized)

    # Tokenize sentence into words
    wordTokenizer = WordPunctTokenizer()
    wordTokens = wordTokenizer.tokenize(inputString)

    # Remove stopwords and Lemmatize
    lemmatizer = WordNetLemmatizer()
    stop = stopwords.words('english')
    stop.append("?")
    wordTokens = [lemmatizer.lemmatize(word) for word in wordTokens if not word in stop]

    # Take set
    vocab = set(wordTokens)

    # Arrange into ascending word
    vocab = sorted(vocab)
    print("Vocabulary len: ", len(vocab))
    vectorize(sentTokenized, vocab)


def vectorize(sentTokens, vocab):
    """
    Takes setence tokens as input
    :param sentTokens:
    :return:
    """

    # print("\n\nVocab : {}, Sentences : {} ".format(len(vocab), len(sentTokens)))
    count = np.zeros((len(sentTokens), len(vocab)), np.float)
    presence = np.zeros((len(sentTokens), len(vocab)), np.float)
    # print(count)

    # Initialize word tokenizer
    wordTokenizer = WordPunctTokenizer()

    # Remove stopwords and lemmatize
    stop = stopwords.words('english')
    stop.append("?")
    lemmatizer = WordNetLemmatizer()

    for j in range(0, len(sentTokens)):
        wordTokens = wordTokenizer.tokenize(sentTokens[j])
        wordTokens = [lemmatizer.lemmatize(word)for word in wordTokens if not word in stop]
        for word in wordTokens:
            index = (vocab.index(word))
            count[j][index] += 1
            presence[j][index] = 1

    # print(count)
    # print(presence)

    tfidf(count, presence, vocab, sentTokens)


def tfidf(count, presence, vocab, sentTokens):
    """
    Computes tf-idf

    :param count:
    :param presence:
    :param vocab:
    :return:
    """

    presence = (presence.sum(axis=0))
    # The number of documents a term is present in.

    # Initialize paramters for TF IDF calculation
    tfidfVec = np.zeros_like(count)
    idfVec = np.zeros_like(presence)
    row, col = len(count), len(count[0])
    totalDocuments = row

    # Compute IDF Vector
    for i in range(0, row):
        for j in range(0, col):
            idfVec[j] = (math.log(totalDocuments / presence[j]) + 1)

    # Compute TF-IDF vector
    for i in range(0, row):
        for j in range(0, col):
            tfidfVec[i][j] = count[i][j] * idfVec[j]

        # Normalize vector
        rootofsquaresum = np.sqrt((tfidfVec[i]*tfidfVec[i]).sum(axis=0))
        tfidfVec[i] = tfidfVec[i]/ rootofsquaresum

    # print("bag of words : " , tfidfVec)
    analyze(tfidfVec, sentTokens, vocab)


def analyze(tfidfVec, sentTokens, vocab, nTopWords = 3):
    """Finds the most important words in a sentence"""
    for vector, sentence in zip(tfidfVec, sentTokens):
        max = np.max(vector)
        sortedVector = sorted(vector)
        sortedVector.reverse()

        # Print n most important words in a sentence
        maxScores = (sortedVector[:nTopWords])
        result = []
        listVector = list(vector)
        for i in range(0, nTopWords):
            index = listVector.index(maxScores[i])
            result.append(vocab[index])
            listVector[index] = 0

        print("\n\n",result,"----", sentence)


if __name__ == "__main__":
    inputString = "How are you doing? I I I guess you have always been doing well."
    input = " Why NSBL Visa Debit Card? What type of cards does Nepal SBI Bank issue?" \
            "How to avail NSBL DEBIT Card? How long will it take to avail the card facility once the application has been submitted to the respective branch?" \
            "How long is the card valid for usage? Shall I get a renewed card against the expired one?" \
            "Can I close the card on my wish? Is there any difference between normal Visa Debit Card and Prepaid Card?" \
            "Where can the cards be used? How much charge I have to pay for new card? " \
            "What if the card gets lost or stolen? Is there any charges on lost card?" \
            "If PIN forgotten, can I get new PIN immediately? Is there any charges for Re-pin generation?" \
            "If card trapped in ATM, how can I get my card? What is the withdrawal limit?" \
            "How much charge to be paid in each withdrawal? Charges differs in each case." \
            "Should I apply for new card if I want to link for a new account?" \
            "Can a customer get a card linked with his/her Overdraft Account?" \
            "Can a joint account holder get NSBL Visa Card? " \
            "I missed to collect cash from the ATM or I tried to withdraw cash from the ATM but machine did not dispense cash where as my account has been debited. In such case what should I do to get my money back?" \
            "Can I use International Cards in Nepal SBI ATM?" \
            "What do the response codes in the ATM transaction slip mean?" \
            "Those are the codes that denote the nature of your transaction. The codes specify whether you were able to successfully complete your transaction or any other error occurred. For a complete list of resp codes and their meaning, read more." \
            "Yes, you may use any international card issued by VISA or MasterCard in Nepal SBI ATMs. The transaction limits and charges for International cards are as follows:" \
            "We require your written complaint. On the basis of your complaint, we will process for it and your account will be credited after we receive chargeback/reimbursement from the concerned bank (ATM). For this we require following details along with your complaint letter" \
            " Joint account which can be operated by any of the account holder can apply for NSBL Visa Debit Card. It can be issued to one or both of the account holders upon their written consent. Whereas, bank reserves the right to accept or reject the request for issuing the card for both account holder." \
            " There is no such provision." \
            " Multiple accounts can be linked in the same card. You need not to apply for a new one   for this purpose. Submit your application to the branch mentioning Your Card No,  Name, Account Number (primary) and the new account number (also mention the type of account to be linked with the card eg. Saving, Current). But bank reserves the right to accept or reject the request." \
            " -If you are doing transaction and balance enquiry on VISA and SBI Group network(other than Nepal SBI Bank) ATMs, the balancing figure will be shown in INR currency" \
            " There is daily and monthly cash withdrawal limit stated as under" \
            " NPR 100 should be paid to re-generate the pin." \
            " A new PIN will be generated and it takes around 25 days to get new PIN. But you have to submit written application to the branch where you have applied for card." \
            " The customer should pay NPR 100.00 for lost card." \
            " Please inform NSBL to block the usage immediately. However, you must send a written application also at the earliest. Please note that you are fully liable for the transactions processed up to the time NSBL is notified of the lost/stolen card. The Bank will issue the replacement of the lost/stolen card upon your written request with applicable charges. However, you cannot get replacement card for Bharat Yatra Card." \
            " For the first time, you have to pay just Rs. 300*/- for normal Card and Rs. 200/- for Prepaid Card. (* subject to change as per account scheme)" \
            " NSBL Visa Debit Card and Prepaid Card can be used in Nepal & India for cash withdrawal as well as purchase." \
            " The difference between normal Visa Debit Card and Prepaid card is that the former has customer name printed in the face of card and the latter has BHARAT YATRA printed in it in the place of customer name. Similarly, for normal debit card, the applicant must maintain account with Nepal SBI Bank , where as for pre-paid card maintaining account is not necessary. Other facilities are same as normal card." \
            " Card can be closed anytime as per the wish of the customer." \
            " Upon the expiry of your card, you can get a renewed card. However, in case of Bharat Yatra card, there is no option for card renewal. New card needs to be purchased once the existing card is expired." \
            " The card is valid for 5 years" \
            "If it is normal Visa Debit Card, it will be ready within 25 days.If it is Prepaid card, you can avail the facility instantly." \
            " To subscribe for NSBL DEBIT CARDS, our account holders can visit any of our branches and submit the duly filled card application form, along with one passport size photo. The bank reserves the right to accept or reject any application. Card application forms are available at all our branches." \
            "We issue domestic Visa Debit Card ,Prepaid domestic cards and USD visa international debit card." \
            ""


    corpusToVocabulary(input)
