import pandas as pd
import os
import numpy as np
from keras.optimizers import Adam
from scipy import sparse as ssp
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from tools import *
from tqdm import tqdm
from io import open
import cPickle as pickle
import json
from gensim.models import KeyedVectors
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Input, \
    Bidirectional, Conv1D, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.models import Model
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence, text
from keras.layers.merge import concatenate, average
from keras.utils.vis_utils import plot_model

from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.utils.data_utils import get_file
from keras import backend as K
import time
from keras_layers import *
from sklearn.utils.class_weight import compute_class_weight
import gc
import datetime

plt.style.use('ggplot')
rootpathdata = '/home/nacim/PycharmProjects/quora_questionpair/data/'
rootpathdata_cleaned = bool_dir(rootpathdata + 'cleaned_data')
rootpathdata_embedding = bool_dir(rootpathdata + 'embedding_data')
rootpathdata_models_weights = bool_dir(rootpathdata + 'models_weights')
TRAIN_DATASET = rootpathdata + 'train.csv'
TEST_DATASET = rootpathdata + 'test.csv'
W2VGLOVE = rootpathdata + 'glove.840B.300d.txt'
W2VGOOGLE = rootpathdata + 'GoogleNews-vectors-negative300.bin.gz'

EMBEDDING_DIM = 300
EMBEDDING_MATRIX_TYPES = ['glove', 'google']
EMBEDDING_MATRIX_FILE = rootpathdata_embedding + '/emmbedding_matrix{0}_{1}.embmat'
TOKENIZER_TYPES = ['only_train', 'train_and_test']
TOKENIZER_FILE = rootpathdata_embedding + '/tokenizer{0}_{1}.token'

NUMBEROFWORDS_FILE = rootpathdata_embedding + '/nb_words{0}_{1}.json'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25

TEST_SPLIT = 0.1
DROPOUT_RATE = 0.2


# def balanced_subsample(x, y, subsample_size=0.165):
#     class_xs = []
#     min_elems = None
#
#     for yi in np.unique(y):
#         elems = x[(y == yi)]
#         class_xs.append((yi, elems))
#         if min_elems == None or elems.shape[0] < min_elems:
#             min_elems = elems.shape[0]
#
#     use_elems = min_elems
#     if subsample_size < 1:
#         use_elems = int(min_elems * subsample_size)
#
#     xs = []
#     ys = []
#
#     for ci, this_xs in class_xs:
#         if len(this_xs) > use_elems:
#             np.random.shuffle(this_xs)
#
#         x_ = this_xs[:use_elems]
#         y_ = np.empty(use_elems)
#         y_.fill(ci)
#
#         xs.append(x_)
#         ys.append(y_)
#
#     xs = np.concatenate(xs)
#     ys = np.concatenate(ys)
#
#     return xs, ys
#
# a = np.random.rand(10)
# ai = np.where(a>0.9)
# print np.sum(ai)
# x = np.arange(0,10)
# print x
# print x[ai]
# exit()

def train_test_split(X, y, test_size=TEST_SPLIT):
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=test_size)
    train, test = [(a, b) for a, b in sss][0]
    return X[train], X[test], y[train], y[test]


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

    if ttype in ['glove', 'google']:
        if questions is None:
            raise ValueError('A list of string is necessary')
        print('Compute Tokenize')
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(questions)
        word_index = tokenizer.word_index

    if ttype == 'glove':
        print ('Use Glove Word2vec')
        embeddings_index = {}
        f = open(W2VGLOVE, encoding='utf-8')
        for line in tqdm(f):
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        nb_words = min(MAX_NB_WORDS, len(word_index))
        embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
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
    # questions = pd.read_csv(os.path.join(rootpathdata_cleaned, 'train_allquestions.csx')).fillna(' ')
    # questions = questions['question'].tolist()
    # init_embedding_matrix(questions=questions, ttype='glove', tokenizertype='only_train', load=True)
    # init_embedding_matrix(questions=questions, ttype='google', tokenizertype='only_train', load=True)
    #
    print('Make embedding matrix only with ALLS  questions')
    #
    questions = pd.read_csv(os.path.join(rootpathdata_cleaned, 'test_allquestions.csx')).fillna(' ')
    questions = questions['question'].tolist()
    #
    init_embedding_matrix(questions=questions, ttype='glove', tokenizertype='train_and_test', load=True)
    # init_embedding_matrix(questions=questions, ttype='google', tokenizertype='train_and_test', load=True)


def build_training_phase_data():
    print('=== LOADING DATA')
    try:
        df = pd.read_csv(os.path.join(rootpathdata_cleaned, 'train.csx'))
        q1_train = df['question1'].tolist()
        # print len(q1_train)
        q2_train = df['question2'].tolist()
        is_duplicate = df['is_duplicate'].tolist()

    except:
        makedata()
        df = pd.read_csv(os.path.join(rootpathdata_cleaned, 'train.csx'))
        q1_train = df['question1'].tolist()
        q2_train = df['question2'].tolist()
        is_duplicate = df['is_duplicate'].tolist()
    labels = np.array(is_duplicate)
    print('=== LOAD TOKENIZER & EMBEDDING MAtRIX')
    tokenizer, embedding_matrix, nb_words = init_embedding_matrix(questions=None, ttype='glove',
                                                                  tokenizertype='only_train', load=True)
    print('===  CREATE SEQUENCES')
    question1_word_sequences = tokenizer.texts_to_sequences(q1_train)
    question2_word_sequences = tokenizer.texts_to_sequences(q2_train)
    q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X = np.stack((q1_data, q2_data), axis=1)
    y = labels
    print('=== MAKE CROSS-VALIDATION SPLIT THAT PRESERVE THE RATIO OF CLASSES IN TRAINING AND VALIDATION SET')
    # X, y = balanced_subsample(X, y)
    # X, _, y, _ = train_test_split(X, y, test_size=0.90)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)

    Q1_train = X_train[:, 0]
    Q2_train = X_train[:, 1]
    # print len(Q1_train)
    # print len(Q2_train)
    # print len(y_train)

    # Q1_trainp = np.concatenate((Q1_train, Q2_train))
    # Q2_trainp = np.concatenate((Q2_train, Q1_train))
    Q1_trainp = Q1_train
    Q2_trainp = Q2_train
    # print len(Q1_trainp)
    # print len(Q2_trainp)
    # y_train = np.concatenate((y_train, y_train))
    # print len(y_train)
    # exit()
    # Compute classes weights
    w1 = np.sum(y_train) / float(len(y_train))
    w0 = 1 - w1
    Q1_test = X_test[:, 0]
    Q2_test = X_test[:, 1]
    # print len(Q1_test)
    # print len(Q2_test)
    validation_data = ([Q1_test, Q2_test], y_test)
    return Q1_trainp, Q2_trainp, embedding_matrix, nb_words, validation_data, w0, w1, y_train


def build_validation_submission_data():
    print('=============== SUBMISSION ===============')
    print('=== LOADING DATA')
    try:
        df = pd.read_csv(os.path.join(rootpathdata_cleaned, 'train.csx'))
        q1_train = df['question1'].tolist()
        q2_train = df['question2'].tolist()
        is_duplicate = df['is_duplicate'].tolist()

        df = pd.read_csv(os.path.join(rootpathdata_cleaned, 'test.csx'))
        q1_sub = df['question1'].tolist()
        q2_sub = df['question2'].tolist()
        testid = df['test_id']
    except:
        makedata()
        df = pd.read_csv(os.path.join(rootpathdata_cleaned, 'train.csx'))
        q1_train = df['question1'].tolist()
        q2_train = df['question2'].tolist()
        is_duplicate = df['is_duplicate'].tolist()
    labels = np.array(is_duplicate)
    print('=== LOAD TOKENIZER & EMBEDDING MAtRIX')
    tokenizer, embedding_matrix, nb_words = init_embedding_matrix(questions=None, ttype='glove',
                                                                  tokenizertype='only_train', load=True)
    print('===  CREATE SEQUENCES')
    question1_word_sequences = tokenizer.texts_to_sequences(q1_train)
    question2_word_sequences = tokenizer.texts_to_sequences(q2_train)
    q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    question1_word_sequences_sub = tokenizer.texts_to_sequences(q1_sub)
    question2_word_sequences_sub = tokenizer.texts_to_sequences(q2_sub)
    q1_data_sub = pad_sequences(question1_word_sequences_sub, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data_sub = pad_sequences(question2_word_sequences_sub, maxlen=MAX_SEQUENCE_LENGTH)
    X = np.stack((q1_data, q2_data), axis=1)
    y = labels
    print('=== REBALANCE TRAINING DATASET')
    # X, y = balanced_subsample(X, y)
    Q1_train = X[:, 0]
    Q2_train = X[:, 1]
    # Q1_train = np.concatenate((Q1_train, Q2_train))
    # print len(Q1_train)
    # Q2_train = np.concatenate((Q2_train, Q1_train))
    # print len(Q2_train)
    # y_train = np.concatenate((y, y))
    # Compute classes weights
    w1 = np.sum(y) / float(len(y))
    w0 = 1 - w1
    submission = ([q1_data_sub, q2_data_sub], testid)
    # SIAMESE_LSTM_WORDEMBEDDING(nb_words=nb_words, word_embedding_matrix=embedding_matrix, nepochs=50, batch_size=512,
    #                            shuffle=True, Q1_train=Q1_train, Q2_train=Q2_train, y_train=y_train,
    #                            class_weight={0: w0, 1: w1},
    #                            validation_data=validation_data)
    return Q1_train, Q2_train, embedding_matrix, nb_words, submission, w0, w1, y


# def SIAMESE_TEMPORAL_DISTRIBUTION_WORDEMBEDDING(nb_words, word_embedding_matrix, nepochs=50, batch_size=1024,
#                                                 shuffle=False,
#                                                 class_weight=None,
#                                                 Q1_train=None, Q2_train=None, y_train=None, validation_split=None,
#                                                 validation_data=None,
#                                                 model_weights_file=os.path.join(rootpathdata_models_weights,
#                                                                                 'trial.h5'), load=False):
#     Q1 = Sequential()
#     Q1.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
#                      trainable=False))
#     Q1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
#     Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,)))
#     Q2 = Sequential()
#     Q2.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
#                      trainable=False))
#     Q2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
#     Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,)))
#     model = Sequential()
#     model.add(Merge([Q1, Q2], mode='concat'))
#     # model.add(BatchNormalization())
#     model.add(Dense(2000, activation='sigmoid', bias_initializer='random_normal'))
#     # model.add(BatchNormalization())
#     # model.add(Dense(400, activation='relu'))
#     # model.add(BatchNormalization())
#     # model.add(Dense(400, activation='relu'))
#     # model.add(BatchNormalization())
#     # model.add(Dense(200, activation=
#     # model.add(BatchNormalization())
#     model.add(Dense(1, activation='sigmoid'))
#
#     model.compile(loss='binary_crossentropy',
#                   optimizer='nadam',
#                   metrics=["accuracy"])
#
#     callbacks = [ModelCheckpoint(model_weights_file, monitor='val_loss', save_best_only=True)]
#
#     # if not os.path.exists(model_weights_file) or not load:
#
#     print("Starting training at", datetime.datetime.now())
#     t0 = time.time()
#     history = model.fit([Q1_train, Q2_train],
#                         y_train,
#                         batch_size=batch_size, shuffle=shuffle,
#                         class_weight=class_weight,
#                         nb_epoch=nepochs,
#                         validation_split=validation_split,
#                         verbose=1,
#                         callbacks=callbacks, validation_data=validation_data)
#     t1 = time.time()
#     # else:
#     #     model.load_weights(model_weights_file)



def BIRNN_ATTENTION_LAYERS(nb_words, word_embedding_matrix, nepochs=50, batch_size=1024, shuffle=True, trainable=False,
                           class_weight=None,
                           Q1_train=None, Q2_train=None, y_train=None, validation_split=None, validation_data=None,
                           model_weights_file=os.path.join(rootpathdata_models_weights, 'trial.h5'), load=False,
                           final_submission=None, test=True, optimizer='adadelta', encode=True, **kwargs):
    """
    Code is based on the paper "A decomposable attention model for natural language inference (2016)" proposed by Aparikh, Oscart, Dipanjand, Uszkoreit. See more detail on https://arxiv.org/abs/1606.01933
    Bahdanau, D., Chorowski, J., Serdyuk, D., Brakel, P., & Bengio, Y. (2016, March). End-to-end attention-based large vocabulary speech recognition. In Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE International Conference on (pp. 4945-4949). IEEE.

    Code is based on the paper "A decomposable attention model for natural language inference (2016)" proposed by Aparikh, Oscart, Dipanjand, Uszkoreit. See more detail on https://arxiv.org/abs/1606.01933

"Reasoning about entailment with neural attention (2016)" proposed by Tim Rockta schel. See more detail on https://arxiv.org/abs/1509.06664

"Neural Machine Translation by Jointly Learning to Align and Translate (2016)" proposed by Yoshua Bengio, Dzmitry Bahdanau, KyungHyun Cho. See more detail on https://arxiv.org/abs/1409.0473
    :return:
    """
    hidden_unit = 100
    # embedding_layer = Embedding(nb_words + 1,
    #                             EMBEDDING_DIM,
    #                             weights=[word_embedding_matrix],
    #                             input_length=MAX_SEQUENCE_LENGTH,
    #                             trainable=trainable,dropout=DROPOUT_RATE,
    #                             nr_tune=5000
    #                             )
    embedding_layer = EmbeddingLayer(nb_words + 1, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, hidden_unit,
                                     init_weights=word_embedding_matrix,
                                     dropout=DROPOUT_RATE, nr_tune=5000)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='words_1')
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='words_2')

    embedded_a = embedding_layer(sequence_1_input)
    embedded_b = embedding_layer(sequence_2_input)

    if encode:
        encoded_a = BiRNN_EncodingLayer(MAX_SEQUENCE_LENGTH, hidden_unit)(embedded_a)
        encoded_b = BiRNN_EncodingLayer(MAX_SEQUENCE_LENGTH, hidden_unit)(embedded_b)
        attention = AttentionLayer(MAX_SEQUENCE_LENGTH, hidden_unit, dropout=DROPOUT_RATE)(encoded_a, encoded_b)
    else:
        attention = AttentionLayer(MAX_SEQUENCE_LENGTH, hidden_unit, dropout=DROPOUT_RATE)(embedded_a, embedded_b)

    align_layer = SoftAlignmentLayer(MAX_SEQUENCE_LENGTH, hidden_unit)
    align_beta = align_layer(embedded_b, attention)  # alignment for sentence a
    align_alpha = align_layer(embedded_a, attention, transpose=True)  # alignment for sentence b

    comp_layer = ComparisonLayer(MAX_SEQUENCE_LENGTH, hidden_unit, dropout=DROPOUT_RATE)
    comp_1 = comp_layer(embedded_a, align_beta)
    comp_2 = comp_layer(embedded_b, align_alpha)

    preds = AggregationLayer(hidden_unit, output_units=1)(comp_1, comp_2)

    # if optimizer == 'adam':
    #     optimizer = Adam(1e-4)

    model = train_model_or_load_train(Q1_train, Q2_train, batch_size, class_weight, load, model_weights_file, nepochs,
                                      optimizer, preds, sequence_1_input, sequence_2_input, validation_data,
                                      y_train)  # model.load_weights(model_weights_file)
    if validation_data is not None:
        test_method(model, validation_data)
    if final_submission is not None:
        submit_score(final_submission, model, model_weights_file)


def BILSTM_WORDEMBEDDING(nb_words, word_embedding_matrix, nepochs=50, batch_size=1024, shuffle=True, trainable=False,
                         class_weight=None,
                         Q1_train=None, Q2_train=None, y_train=None, validation_split=None, validation_data=None,
                         model_weights_file=os.path.join(rootpathdata_models_weights, 'trial.h5'), load=False,
                         final_submission=None, test=True, optimizer='adadelta', **kwargs):
    """

    :return:
    """
    print("=== INITIALIZE THE LSTM MODEL with word")

    num_lstm = 175  # np.random.randint(175, 275)
    num_dense = 64  # np.random.randint(100, 150)
    rate_drop_lstm = 0.25
    rate_drop_dense = 0.5

    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[word_embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=trainable)

    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm,  # bias_initializer='random_normal',
                      recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    m = concatenate([x1, y1])
    m = Dropout(rate_drop_dense)(m)
    m = BatchNormalization()(m)

    m = Dense(num_dense, activation='relu')(m)
    m = Dropout(rate_drop_dense)(m)
    m = BatchNormalization()(m)

    preds = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(m)

    model = train_model_or_load_train(Q1_train, Q2_train, batch_size, class_weight, load, model_weights_file, nepochs,
                                      optimizer, preds, sequence_1_input, sequence_2_input, validation_data,
                                      y_train)  # model.load_weights(model_weights_file)
    if validation_data is not None:
        test_method(model, validation_data)
    if final_submission is not None:
        submit_score(final_submission, model, model_weights_file)


def train_model_or_load_train(Q1_train, Q2_train, batch_size, class_weight, load, model_weights_file, nepochs,
                              optimizer, preds, sequence_1_input, sequence_2_input, validation_data, y_train):
    # Train the model
    y_train, Q1_train, Q2_train = unison_shuffled_copies(y_train, Q1_train, Q2_train)
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    print ('start compiling', datetime.datetime.now())
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

    plot_model(model, model_weights_file + '__plot_graph.png', show_shapes=True, show_layer_names=False)
    # exit()
    early_stopping = EarlyStopping(monitor='val_loss', patience=25)
    model_checkpoint = ModelCheckpoint(model_weights_file + '.h5', save_best_only=True)
    if load and os.path.exists(model_weights_file):
        print ('Load  already trained weights from the file %s' % model_weights_file)
        model.load_weights(model_weights_file + '.h5')
        hist = model.fit([Q1_train, Q2_train], y_train,  # validation_data=validation_data,
                         validation_split=0.2,
                         epochs=nepochs, batch_size=batch_size, shuffle=True,
                         class_weight=class_weight, verbose=1)
        plot_training(hist, model_weights_file + '_continue')

    else:
        hist = model.fit([Q1_train, Q2_train], y_train, validation_data=validation_data,
                         validation_split=0.2,
                         epochs=nepochs, batch_size=batch_size, shuffle=True,
                         class_weight={0: 0.79264156344230219, 1: 1.3542873987525375},
                         callbacks=[early_stopping, model_checkpoint], verbose=1)

        model.save_weights(model_weights_file + '.h5', overwrite=True)

        bst_val_score = min(hist.history['val_loss'])
        plot_training(hist, model_weights_file)
    model.load_weights(model_weights_file + '.h5')
    return model


def unison_shuffled_copies(y, q1, q2):
    assert len(y) == len(q1) == len(q2)
    p = np.random.permutation(len(y))
    return y[p], q1[p], q2[p]


def plot_training(hist, model_weights_file):
    plt.plot(hist.history['acc'], label='accuracy')
    plt.plot(hist.history['val_acc'], label='val accurayc')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(model_weights_file + '_plot_training.pdf', bbox_inches='tight')


def test_method(model, validation_data):
    print('Test Solution')
    [Q1_test, Q2_test], y_test = validation_data
    ypred = model.predict([Q1_test, Q2_test], batch_size=8192, verbose=1).ravel()
    score = log_loss(y_test, ypred)
    print score


def submit_score(final_submission, model, model_weights_file):
    print("=== MAKE THE SUBMISSION")
    # Create the submission file
    [Q1_test, Q2_test], test_id = final_submission
    ypred = model.predict([Q1_test, Q2_test], batch_size=8192, verbose=1).ravel()
    submit = pd.DataFrame()
    submit['test_id'] = test_id
    submit['is_duplicate'] = ypred
    submit.to_csv(model_weights_file + 'submission.csv', index=False)
    print('Writing the submission file...')


# def CONV1D_EMBEDDING(nb_words, word_embedding_matrix, nepochs=50, batch_size=1024, shuffle=False, trainable=False,
#                      class_weight=None,
#                      Q1_train=None, Q2_train=None, y_train=None, validation_split=None, validation_data=None,
#                      model_weights_file=os.path.join(rootpathdata_models_weights, 'trial.h5',optimizer='adadelta'), load=False,
#                      final_submission=None, test=True, ):
#     """
#
#     :return:
#     """
#     print("=== INITIALIZE THE CONV1D MODEL with word")
#
#     num_lstm = 150  # np.random.randint(175, 275)
#     num_dense = 128# np.random.randint(100, 150)
#     rate_drop_lstm = 0.15 + np.random.rand() * 0.25
#     rate_drop_dense = 0.15 + np.random.rand() * 0.25
#     num_filters = 128
#     filter_sizes = [5]
#
#     rate_drop_cnn = 0.5
#     rate_drop_dense = 0.15 + np.random.rand() * 0.25
#     #
#     # Define the model structure
#     # ----------------------------------------------------------------------------
#     embedding_layer = Embedding(nb_words+1,
#                             EMBEDDING_DIM,
#                             weights=[word_embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)
#     lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm,  # bias_initializer='random_normal',
#                       recurrent_dropout=rate_drop_lstm)
#
#     sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#     embedded_sequences_1 = embedding_layer(sequence_1_input)
#
#     conv_blocks = []
#     for ks in filter_sizes:
#         conv = Conv1D(filters=num_filters,
#                       kernel_size=ks,
#                       padding='valid',
#                       activation="relu",
#                       strides=1)(embedded_sequences_1)
#         conv = MaxPooling1D(pool_size=2)(conv)
#         # conv = Flatten()(conv)
#         conv_blocks.append(conv)
#     convs1 = average(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
#     convs1 = lstm_layer(convs1)
#
#     sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#     embedded_sequences_2 = embedding_layer(sequence_2_input)
#
#     conv_blocks = []
#     for ks in filter_sizes:
#         conv2 = Conv1D(filters=num_filters,
#                        kernel_size=ks,
#                        padding='valid',
#                        activation="relu",
#                        strides=1)(embedded_sequences_2)
#         conv2 = MaxPooling1D(pool_size=2)(conv2)
#         # conv2 = Flatten()(conv2)
#         conv_blocks.append(conv2)
#     convs2 = average(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
#     convs2 = lstm_layer(convs2)
#
#     merged = concatenate([convs1, convs2])
#     merged = Conv1D(filters=256,kernel_size=ks,padding='valid',activation='relu')(merged)
#     merged = MaxPooling1D()(merged)
#
#     merged = lstm_layer(merged)
#
#     # merged = Dropout(rate_drop_dense)(merged)
#
#     merged = Dense(num_dense, activation='sigmoid')(merged)
#     merged = Dropout(rate_drop_dense)(merged)
#     merged = BatchNormalization()(merged)
#
#     merged = Dense(num_dense, activation='sigmoid')(merged)
#     merged = Dropout(rate_drop_dense)(merged)
#     merged = BatchNormalization()(merged)
#     merged = Flatten()(merged)
#     preds = Dense(1, activation='sigmoid')(merged)
#
#     #
#     # Train the model
#     # ----------------------------------------------------------------------------
#     model = Model(inputs=[sequence_1_input, sequence_2_input],
#                   outputs=preds)
#     model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['acc'])
#
#     print ('start compiling', datetime.datetime.now())
#
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#     model_checkpoint = ModelCheckpoint(model_weights_file, save_best_only=True)
#
#     if load and os.path.exists(model_weights_file):
#         print ('Load  already trained weights from the file %s' % model_weights_file)
#         model.load_weights(model_weights_file)
#
#         hist = model.fit([Q1_train, Q2_train], y_train,  # validation_data=validation_data,
#                          validation_split=0.2,
#                          epochs=nepochs, batch_size=batch_size, shuffle=True,
#                          class_weight=class_weight, verbose=1)
#
#     else:
#         hist = model.fit([Q1_train, Q2_train], y_train, validation_data=validation_data,
#                          # validation_split=0.2,
#                          epochs=nepochs, batch_size=batch_size, shuffle=True,
#                          class_weight=class_weight, callbacks=[early_stopping, model_checkpoint], verbose=1)
#
#         model.save_weights(model_weights_file, overwrite=True)
#
#         bst_val_score = min(hist.history['val_loss'])
#         plt.plot(hist.history['acc'], label='accuracy')
#         plt.plot(hist.history['val_acc'], label='val accurayc')
#         plt.title('model accuracy')
#         plt.ylabel('accuracy')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'test'], loc='upper left')
#         # summarize history for loss
#         plt.plot(hist.history['loss'], label='loss')
#         plt.plot(hist.history['val_loss'], label='val loss')
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend()
#         plt.savefig(model_weights_file + '.pdf', bbox_inches='tight')
#
#     # model.load_weights(model_weights_file)
#
#     if validation_data is not None:
#         print('Test Solution')
#         [Q1_test, Q2_test], y_test = validation_data
#         ypred = model.predict([Q1_test, Q2_test], batch_size=8192, verbose=1).ravel()
#         ypred2 = model.predict([Q2_test, Q1_test], batch_size=8192, verbose=1).ravel()
#         final = (ypred + ypred2) / 2.
#         score = log_loss(y_test, ypred)
#         print score
#         score = log_loss(y_test, final)
#         print score
#
#     if final_submission is not None:
#         """
#         """
#         print("=== MAKE THE SUBMISSION")
#         # Create the submission file
#         [Q1_test, Q2_test], test_id = final_submission
#         ypred = model.predict([Q1_test, Q2_test], batch_size=8192, verbose=1).ravel()
#         submit = pd.DataFrame()
#         submit['test_id'] = test_id
#         submit['is_duplicate'] = ypred
#         submit.to_csv(model_weights_file + 'submission.csv', index=False)
#
#         print('Writing the submission file...')



def xp_training(ttype_token='only_train', ttype_w2v="google", model_weights_file='test.h5', algoname='bilstm',
                optimizer='adadelta', encode=True):
    """

    :return:
    """

    if 'conv' in algoname:
        conv = True
    else:
        conv = False
    Q1_train, Q2_train, embedding_matrix, nb_words, validation_data, w0, w1, y_train = build_training_phase_data()

    if algoname == 'bilstm':
        algo = BILSTM_WORDEMBEDDING
    elif algoname == 'conv1D':
        algo = CONV1D_EMBEDDING
    elif algoname == 'birnnattention':
        algo = BIRNN_ATTENTION_LAYERS

    weights = {}
    weights[0], weights[1] = compute_class_weight('balanced', [0, 1], y_train)
    # print weights
    # exit()
    algo(nb_words=nb_words, word_embedding_matrix=embedding_matrix, nepochs=100,
         batch_size=1024, model_weights_file=model_weights_file,
         shuffle=True, Q1_train=Q1_train, Q2_train=Q2_train, y_train=y_train,
         # class_weight={0: w0, 1: w1},
         class_weight=weights,
         test=True,
         validation_data=validation_data, trainable=False, final_submission=None,
         load=False, optimizer=optimizer)


def xp_submit(ttype_token='only_train', ttype_w2v="google", model_weights_file='test.h5', optimizer='adadelta',
              encode=False):
    """

    :return:
    """
    if 'conv' in algoname:
        conv = True
    else:
        conv = False

    Q1_train, Q2_train, embedding_matrix, nb_words, submission, w0, w1, y_train = build_validation_submission_data()
    if algoname == 'bilstm':
        algo = BILSTM_WORDEMBEDDING
    elif algoname == 'conv1D':
        algo = CONV1D_EMBEDDING
    elif algoname == 'birnnattention':
        algo = BIRNN_ATTENTION_LAYERS
    weights = {}
    weights[0], weights[1] = compute_class_weight('balanced', [0, 1], y_train)
    algo(nb_words=nb_words, word_embedding_matrix=embedding_matrix, nepochs=5,
         batch_size=1024, model_weights_file=model_weights_file,
         shuffle=True, Q1_train=Q1_train, Q2_train=Q2_train, y_train=y_train,
         class_weight=weights, test=True,
         validation_data=None, trainable=False, final_submission=submission,
         load=True, optimizer=optimizer, encode=encode)


if __name__ == '__main__':
    if 1 > 3:
        makedata()
    ttype_token = 'only_train'
    ttype_w2v = "google"
    algoname = 'birnnattention'
    optimizer = 'adam'
    encode = True
    info = ''
    if encode and algoname == 'birnnattention':
        info = '_enc'

    model_weights_file = os.path.join(rootpathdata_models_weights,
                                      '{2}_{0}_{1}_{3}{4}'.format(ttype_token, ttype_w2v, algoname, optimizer, info))
    # model_weights_file = os.path.join(rootpathdata_models_weights, 'bi.h5')
    xp_training(ttype_token='only_train', ttype_w2v="glove", model_weights_file=model_weights_file, algoname=algoname,
                optimizer=optimizer, encode=encode)
    xp_submit(ttype_token='only_train', ttype_w2v="glove", model_weights_file=model_weights_file, optimizer=optimizer,
              encode=encode)
