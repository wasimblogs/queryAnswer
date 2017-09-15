import gzip
import bigramPreprocess


def parseQA(path, debug=False):
    """
    parse file
    :param path:
    :param debug: print intermediate values to help debugging
    :return: questions, answers
    """
    readZip = gzip.open(path, 'r')

    questions = []
    answers = []
    for qa in readZip:
        qa = eval(qa)
        question = qa['question']
        answer = qa['answer']
        questions.append(question.lower())
        answers.append(answer.lower())

        if debug == True:
            print(qa.keys())
            break

            # print(question, "\t\t","\n",answer)
            # print("--"*20)

    return questions, answers


def parseMultipleAnswerQuestion(path):
    g = gzip.open(path, 'r')

    count = 0
    for l in g:
        a = (eval(l))
        qa = a['questions'][0]
        question = (qa['questionText'])
        print(count, question)
        items = (qa['answers'])
        for item in items:
            print(item['answerText'])

        print("*" * 40)
        print("\n\n")
        count += 1


filename = "Dataset\qa_Musical_Instruments.json.gz"
questions, answers = parseQA(filename)
# vocab, _ = bigramPreprocess.bigramVocabularyFromCorpus(questions, debug=True)
bigramPreprocess.buildQAModel(questions, True, unigram=True)
