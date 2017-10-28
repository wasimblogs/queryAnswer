"""
Supervised learning for finding duplicate questions on Quora Database
"""

import textProcessing as tp


def readQuoraDB(n=80, insertSyn=False, computeBigram=True):
    """
    n = 5 ideal for debugging logical flaws
    :param n: 
    :param insertSyn: 
    :param computeBigram: 
    :return: 
    """

    # Read quora db and split lines
    db = open("quora_duplicate_questions.tsv", "r", encoding='utf-8').read()
    db = db.split("\n")
    db.pop(0)

    questionList1 = []
    questionList2 = []
    for index, row in enumerate(db):
        row = row.split('\t')
        if len(row) == 6:
            _, _, _, q1, q2, duplicate = row
            if not int(duplicate):
                questionList1.append(q1)
                questionList2.append(q2)

    dbQueries = questionList1[:n]  # Queries stored in database
    userQueries = questionList2[:n]  # Supposedly asked by user

    print("No of questions : ", len(dbQueries))
    print("Processing DB Query")

    print("1/4. Finding vocab")
    vocab = tp.vocabFromCorpus(dbQueries, showDebugInfo=False, computeBigram=computeBigram, minUnigramFreq=0)
    print("2/4. Tokenizing Sentence")
    tokenizedSents = tp.tokenizeIntoWords(dbQueries, vocab=vocab, showDebugInfo=False,
                                                   computeBigram=computeBigram)
    print("3/4. Vectorizing Sentence")
    count, presence = tp.vectorize(tokenizedSents, vocab, showDebugInfo=False)
    print("4/4. TF IDF calculation")
    tfidfVecDB = tp.tfidf(count=count, presence=presence, vocab=vocab)

    # # Checking top keywords in a sentence
    # for index, sents in enumerate(tokenizedSents):
    #     print(dbQueries[index])
    #     # print(tfidfVecDB[index])
    #     result = tp.nTopWordsinSentence(tfidfVecDB[index], sentTokens=sents, vocab=vocab, n=10)
    #     print("Keyword: ", result, "\n\n")

    accuracy = 0
    for index, query in enumerate(userQueries):
        # print("\n\n", "*" * 60, "\n\n", index, "Question : ", query)

        processedQuery = tp.processQuery(query, vocab, computeBigram=computeBigram, insertSynonym=insertSyn,
                                                  showDebugInfo=False)
        queryVec = tp.vectorizeUserQuery(processedQuery, vocab, showDebugInfo=False)
        nKeywords = tp.nTopWordsinSentence(queryVec, sentTokens=processedQuery, vocab=vocab, n=10,
                                                    showDebugInfo=False)

        queryMatchScore = tp.bestMatchForQuery(queryVec, tfidfVecDB, 0,
                                                           topNMatch=1, considerAnswer=False)
        score, indexDB = queryMatchScore[0]

        if index == indexDB:
            accuracy += 1
            # print("What I got right ", "/" * 20)
            # print(indexDB, "score: %0.2f" % score, dbQueries[indexDB])

        else:
            print("\n\n", "*" * 60, "\n\n", index, "Question : ", query)
            print("Duplicate: ", questionList1[index])
            print("Keywords : ", nKeywords)

            print("What I got wrong", "*" * 20)
            print(indexDB, "score: %0.2f" % score, dbQueries[indexDB])
            pass

    print(n, accuracy, "\tAccuracy : ", accuracy / n)


if __name__ == "__main__":
    readQuoraDB()
    pass
