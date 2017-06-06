import pandas as pd
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from deep_learning.tools import train_test_split
from tools import *
from tqdm import tqdm
from io import open
import cPickle as pickle
import json
from keras.preprocessing.text import Tokenizer
from gensim.models import KeyedVectors

STOPS = set(stopwords.words('english'))

ROOTHPATHDATA = os.path.join(os.path.expanduser('~'), 'DATASET_KAGGLE/quora')
rootpathdata_cleaned = bool_dir(os.path.join(ROOTHPATHDATA, 'cleaned_data'))
rootpathdata_embedding = bool_dir(os.path.join(ROOTHPATHDATA, 'embedding_data'))
rootpathdata_models_weights = bool_dir(os.path.join(ROOTHPATHDATA, 'models_weights'))

TRAIN_DATASET = os.path.join(ROOTHPATHDATA, 'train.csv')
TEST_DATASET = os.path.join(ROOTHPATHDATA, 'test.csv')
W2VGLOVE = os.path.join(ROOTHPATHDATA, 'glove.840B.300d.txt')
W2VGOOGLE = os.path.join(ROOTHPATHDATA, 'GoogleNews-vectors-negative300.bin.gz')
W2VTWITTER = os.path.join(ROOTHPATHDATA, 'glove.twitter.27B.50d.txt')

EMBEDDING_DIM = 300
EMBEDDING_DIM_TWITTER = 50
EMBEDDING_MATRIX_TYPES = ['glove', 'google']
EMBEDDING_MATRIX_FILE = os.path.join(rootpathdata_embedding, 'emmbedding_matrix{0}_{1}.embmat')
TOKENIZER_TYPES = ['only_train', 'train_and_test']
TOKENIZER_FILE = os.path.join(rootpathdata_embedding, 'tokenizer{0}_{1}.token')

NUMBEROFWORDS_FILE = os.path.join(rootpathdata_embedding, 'nb_words{0}_{1}.json')
MAX_NB_WORDS = 50000
PATH_TEST_DATASET = os.path.join(rootpathdata_cleaned, 'test.csx')
PATH_TRAIN_DATASET = os.path.join(rootpathdata_cleaned, 'train.csx')


# To clean text
def cleanText(t):
    # print t
    # Make lower case
    # t = unicode(t,errors='replace')
    t = t.str.lower()
    # Remove all characters that are not in the defined alphabet
    # Final alphabet : [a-z0-9!?:'$%^*+-= ]
    t = t.str.replace(r"[^a-z0-9!?:'$%^&*+-= ]", " ")
    # Clean text
    t = t.str.replace(r"+", " ")
    t = t.str.replace(r" & ", " and ")
    t = t.str.replace(r" &", " and ")
    t = t.str.replace(r"& ", " and ")
    t = t.str.replace(r"&", " and ")
    t = t.str.replace(r"what's", "what is")
    t = t.str.replace(r"'s", "")
    t = t.str.replace(r"'ve", " have")
    t = t.str.replace(r"can't", "cannot")
    t = t.str.replace(r"n't", " not")
    t = t.str.replace(r"i'm", "i am")
    t = t.str.replace(r"'re", " are")
    t = t.str.replace(r"'d", " would")
    t = t.str.replace(r"'ll", " will")
    t = t.str.replace(r"'", "")
    t = t.str.replace(r"(\d+)(k)", r"\g<1>000")
    t = t.str.replace(r" e g ", " eg ")
    t = t.str.replace(r" b g ", " bg ")
    t = t.str.replace(r" u s ", " american ")
    t = t.str.replace(r"0s", "0")
    t = t.str.replace(r" 9 11 ", " 911 ")
    t = t.str.replace(r"e - mail", "email")
    t = t.str.replace(r"j k", "jk")
    t = t.str.replace(r"\s{2,}", " ")
    t = t.str.replace(r"what's", "what is ")
    t = t.str.replace(r"\'s", " ")
    t = t.str.replace(r"\'ve", " have ")
    t = t.str.replace(r"can't", "cannot ")
    t = t.str.replace(r"n't", " not ")
    t = t.str.replace(r"60k", " 60000 ")
    t = t.str.replace(r" u s ", " american ")
    t = t.str.replace(r" 9 11 ", "911")
    t = t.str.replace(r"quikly", "quickly")
    t = t.str.replace(r"usa", "America")
    t = t.str.replace(r"canada", "Canada")
    t = t.str.replace(r"japan", "Japan")
    t = t.str.replace(r"germany", "Germany")
    t = t.str.replace(r"burma", "Burma")
    t = t.str.replace(r"rohingya", "Rohingya")
    t = t.str.replace(r"zealand", "Zealand")
    t = t.str.replace(r"cambodia", "Cambodia")
    t = t.str.replace(r"zealand", "Zealand")
    t = t.str.replace(r"norway", "Norway")
    t = t.str.replace(r" uk ", " England ")
    t = t.str.replace(r"india", "India")
    t = t.str.replace(r"pakistan", "Pakistan")
    t = t.str.replace(r"britain", "Britain")
    t = t.str.replace(r"switzerland", "Switzerland")
    t = t.str.replace(r"china", "China")
    t = t.str.replace(r"chinese", "Chinese")
    t = t.str.replace(r"imrovement", "improvement")
    t = t.str.replace(r"intially", "initially")
    t = t.str.replace(r"quora", "Quora")
    t = t.str.replace(r" dms ", "direct messages ")
    t = t.str.replace(r"demonitization", "demonetization")
    t = t.str.replace(r"actived", "active")
    t = t.str.replace(r"kms", " kilometers ")
    t = t.str.replace(r" cs ", " computer science ")
    t = t.str.replace(r" upvotes ", " up votes ")
    t = t.str.replace(r" iphone ", " phone ")
    # t = t.str.replace(r"\0rs ", " rs ")
    t = t.str.replace(r"calender", "calendar")
    t = t.str.replace(r"ios", "operating system")
    t = t.str.replace(r"gps", "GPS")
    t = t.str.replace(r"gst", "GST")
    t = t.str.replace(r"programing", "programming")
    t = t.str.replace(r"bestfriend", "best friend")
    t = t.str.replace(r"dna", "DNA")
    t = t.str.replace(r"iii", "3")
    t = t.str.replace(r"california", "California")
    t = t.str.replace(r"texas", "Texas")
    t = t.str.replace(r"tennessee", "Tennessee")
    t = t.str.replace(r"the us", "America")
    t = t.str.replace(r"trump", "Trump")
    return t


def init_embedding_matrix(questions=None, ttype="glove", tokenizertype='only_train', load=False):
    ffile_embedmatrix = EMBEDDING_MATRIX_FILE.format(ttype, tokenizertype)
    ffilenmwords = NUMBEROFWORDS_FILE.format(ttype, tokenizertype)
    ffilentoken = TOKENIZER_FILE.format(ttype, tokenizertype)
    if load:
        if os.path.exists(ffile_embedmatrix) and os.path.exists(ffilenmwords) and os.path.exists(ffilentoken):
            print('Loading tokenizer:{0} with embedding matrix  made with {1} Word2Vec and the number of words'.format(
                tokenizertype, ttype))
            with open(ffilentoken, 'rb') as f:
                tokenizer = pickle.load(f)
            with open(ffilenmwords, 'r') as f:
                nb_words = json.load(f)['nb_words']
            embedding_matrix = np.load(open(ffile_embedmatrix, 'rb'))
            return tokenizer, embedding_matrix, nb_words
        else:
            pass
        print('Sorry no Files, or there are missing ones')

    if ttype in ['glove', 'google', 'twitter']:
        if questions is None:
            raise ValueError('A list of string is necessary')
        print('Compute Tokenize')
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(questions)
        word_index = tokenizer.word_index

    if ttype == 'glove' or ttype=='twitter':
        print ('Use Glove Word2vec'+' with twitter')
        embeddings_index = {}
        if ttype=='twitter':

            f = open(W2VTWITTER, encoding='utf-8')
        else:
            f = open(W2VGLOVE, encoding='utf-8')
        for line in tqdm(f):
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        dim = EMBEDDING_DIM_TWITTER if ttype=='twitter' else EMBEDDING_DIM
        nb_words = min(MAX_NB_WORDS, len(word_index))
        embedding_matrix = np.zeros((nb_words + 1, dim))
        for word, i in word_index.items():
            if i > MAX_NB_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    elif ttype == 'google':
        print ('Use Google Word2vec')
        model = KeyedVectors.load_word2vec_format(W2VGOOGLE, binary=True)
        nb_words = min(MAX_NB_WORDS, len(word_index))
        embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i > MAX_NB_WORDS:
                continue
            if word in model.vocab:
                embedding_matrix[i] = model.word_vec(word)

    try:
        np.save(open(ffile_embedmatrix, mode='wb'), embedding_matrix)
        with open(ffilenmwords, 'w', encoding="utf-8") as f:
            r = {'nb_words': nb_words}
            f.write(unicode(json.dumps(r, ensure_ascii=False)))

        with open(ffilentoken, 'wb') as f:
            pickle.dump(tokenizer, f)

    except:
        pass

    return tokenizer, embedding_matrix, nb_words


def makedata():
    """
    Make Dataset with separate questions after cleaning them
    :return:
    """
    # train
    print('Clean Train Dataset and separate questions')
    df = pd.read_csv(TRAIN_DATASET).replace(np.nan, ' ')
    t = df.shape[0] * 2
    print t
    df['question1'] = cleanText(df['question1'])
    df['question2'] = cleanText(df['question2'])

    df.to_csv(os.path.join(rootpathdata_cleaned, 'train.csx'), index=False)
    overallquestions = df['question1'].tolist() + df['question2'].tolist()
    tpm = pd.DataFrame()
    tpm['question'] = overallquestions
    tpm.to_csv(os.path.join(rootpathdata_cleaned, 'train_allquestions.csx'), index=False)
    # test

    print('Clean Test Dataset and separate questions')
    df = pd.read_csv(TEST_DATASET).fillna(' ')
    t1 = df.shape[0] * 2
    df['question1'] = cleanText(df['question1'])
    df['question2'] = cleanText(df['question2'])
    df.to_csv(os.path.join(rootpathdata_cleaned, 'test.csx'), index=False)

    overallquestions += df['question1'].tolist() + df['question2'].tolist()
    tpm = pd.DataFrame()
    tpm['question'] = overallquestions
    tpm.to_csv(os.path.join(rootpathdata_cleaned, 'test_allquestions.csx'), index=False)
    print len(overallquestions), t1 + t


def make_embeddingmatrix():
    print('Make embedding matrix only with questions of the training dataset')
    questions = pd.read_csv(os.path.join(rootpathdata_cleaned, 'train_allquestions.csx')).fillna(' ')
    questions = questions['question'].tolist()
    init_embedding_matrix(questions=questions, ttype='glove', tokenizertype='only_train', load=True)
    init_embedding_matrix(questions=questions, ttype='google', tokenizertype='only_train', load=True)
    init_embedding_matrix(questions=questions, ttype='twitter', tokenizertype='only_train', load=True)
    #
    # print('Make embedding matrix only with ALLS  questions')
    # #
    # questions = pd.read_csv(os.path.join(rootpathdata_cleaned, 'test_allquestions.csx')).fillna(' ')
    # questions = questions['question'].tolist()
    # #
    # init_embedding_matrix(questions=questions, ttype='glove', tokenizertype='train_and_test', load=True)
    # init_embedding_matrix(questions=questions, ttype='google', tokenizertype='train_and_test', load=True)


def build_validation_submission_data(tokenizer, embedding_matrix, nb_words, maxsequencelength):
    print('=============== SUBMISSION ===============')
    try:

        df = pd.read_csv(PATH_TEST_DATASET)
        q1_sub = df['question1'].tolist()
        q2_sub = df['question2'].tolist()
        testid = df['test_id']
    except:
        print('No TEST DATASET')

    print('===  CREATE SEQUENCES')
    question1_word_sequences_sub = tokenizer.texts_to_sequences(q1_sub)
    question2_word_sequences_sub = tokenizer.texts_to_sequences(q2_sub)
    q1_data_sub = pad_sequences(question1_word_sequences_sub, maxlen=maxsequencelength)
    q2_data_sub = pad_sequences(question2_word_sequences_sub, maxlen=maxsequencelength)
    submission = ([q1_data_sub, q2_data_sub], testid)
    return submission


def build_training_phase_data(validation, duplicate, ttype, tokentype, maxsequencelength):
    print('=== LOADING TRAINING DATA')
    try:
        df = pd.read_csv(PATH_TRAIN_DATASET)
        q1_train = df['question1'].tolist()
        # print len(q1_train)
        q2_train = df['question2'].tolist()
        is_duplicate = df['is_duplicate']

    except:
        makedata()
        df = pd.read_csv(PATH_TRAIN_DATASET)
        q1_train = df['question1'].tolist()
        q2_train = df['question2'].tolist()
        is_duplicate = df['is_duplicate']
    print('=== LOAD TOKENIZER & EMBEDDING MAtRIX')
    tokenizer, embedding_matrix, nb_words = init_embedding_matrix(questions=None, ttype=ttype,
                                                                  tokenizertype=tokentype, load=True)
    print('===  CREATE SEQUENCES')
    question1_word_sequences = tokenizer.texts_to_sequences(q1_train)
    question2_word_sequences = tokenizer.texts_to_sequences(q2_train)
    q1_data = pad_sequences(question1_word_sequences, maxlen=maxsequencelength)
    q2_data = pad_sequences(question2_word_sequences, maxlen=maxsequencelength)
    X_train = np.stack((q1_data, q2_data), axis=1)
    y_train = np.array(is_duplicate)

    yi = np.where(y_train==1)
    y_train[yi] = -1
    yi = np.where(y_train==0)
    y_train[yi] = 1
    yi = np.where(y_train == -1)
    y_train[yi] = 0

    print('=== MAKE CROSS-VALIDATION SPLIT THAT PRESERVE THE RATIO OF CLASSES IN TRAINING AND VALIDATION SET')
    # X, y = balanced_subsample(X, y)
    # X, _, y, _ = train_test_split(X, y, test_size=0.90)
    if validation:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    Q1_train = X_train[:, 0]
    Q2_train = X_train[:, 1]

    if duplicate:
        Q1_trainp = np.concatenate((Q1_train, Q2_train))
        Q2_trainp = np.concatenate((Q2_train, Q1_train))
        y_train = np.concatenate((y_train, y_train))

    else:
        Q1_trainp = Q1_train
        Q2_trainp = Q2_train
    validation_data = None
    if validation:
        Q1_test = X_test[:, 0]
        Q2_test = X_test[:, 1]
        validation_data = ([Q1_test, Q2_test], y_test)

    return Q1_trainp, Q2_trainp, y_train, embedding_matrix, nb_words, tokenizer, validation_data


if __name__ == '__main__':
    print('Make sure to change the ROOTPATHDATA')
    print ("The current direction of data is %s" % ROOTHPATHDATA)
    if not os.path.exists(TRAIN_DATASET):
        assert IOError(
            'Please Copy the training dataset available at this link: https://www.kaggle.com/c/quora-question-pairs and unzip in the root folder you chosen')
    if not os.path.exists(TRAIN_DATASET):
        assert IOError(
            'Please Copy the test dataset available at this link: https://www.kaggle.com/c/quora-question-pairs and unzip in the root folder you chosen')

    makedata()
    make_embeddingmatrix()
