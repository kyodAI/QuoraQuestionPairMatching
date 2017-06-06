import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, SpatialDropout1D
from sklearn.metrics import log_loss
import tensorflow as tf
from deep_learning.data import build_validation_submission_data, build_training_phase_data, rootpathdata_cleaned
from keras_layers import *
from tools import *
from keras.models import Model
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
# from keras.activations import relu
from keras.layers.noise import GaussianNoise

XP_FOLDER = bool_dir(os.path.join(os.path.curdir, 'experiments'))

MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300
DROPOUT_RATE = 0.2
TRAIN_FILE = rootpathdata_cleaned


class DEEPNETWORKS(object):
    def __init__(self, name):
        """

        """
        if name == 'bilstm':
            pass
            self.algo = self.BILSTM
        elif name == 'conv1D':
            pass
            self.algo = self.BICONV1DNN
        elif name == 'birnnattention':
            pass
            self.algo = self.BIRNN_ATTENTION_LAYERS
        elif name == 'biconv1dlstm':
            self.algo = self.BICONVLSTM
        elif name == 'biconv1d_mergelstm':
            self.algo = self.BICONV_MERGELSTM

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        loss, optimizer, preds, sequence_1_input, sequence_2_input = self.algo(*args, **kwargs)

        return self.build_model(loss, optimizer, preds, sequence_1_input, sequence_2_input)

    def build_model(self, loss, optimizer, preds, sequence_1_input, sequence_2_input):
        model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
        print("Model is compiling: %s" % get_time())
        model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
        return model

    def test_method(self, model, validation_data, xpfile):
        print('Test Solution')
        [Q1_test, Q2_test], y_test = validation_data
        ypred = model.predict([Q1_test, Q2_test], batch_size=8192, verbose=1).ravel()
        score = log_loss(y_test, ypred)
        d = pd.DataFrame()
        d['true'] = y_test
        d['pred'] = ypred
        d.to_csv(xpfile + '_validation.csv', index=False)

    def submit_score(self, final_submission, model, model_weights_file):
        print("=== MAKE THE SUBMISSION")
        # Create the submission file
        [Q1_test, Q2_test], test_id = final_submission
        ypred = model.predict([Q1_test, Q2_test], batch_size=8192, verbose=1).ravel()
        submit = pd.DataFrame()
        submit['test_id'] = test_id
        submit['is_duplicate'] = ypred
        submit.to_csv(model_weights_file + 'submission.csv', index=False)
        print('Writing the submission file...')

    def train(self, model, Q1_train, Q2_train, y_train, batch_size, xpfile, nepochs):
        # Train the model
        y_train, Q1_train, Q2_train = unison_shuffled_copies(y_train, Q1_train, Q2_train)

        plot_model(model, xpfile + '_plot_model.png', show_shapes=True, show_layer_names=False)
        # exit()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint(xpfile + '.h5', save_best_only=True)
        # learningrate = AdaptiveLearningRateScheduler(1e-4)
        hist = model.fit([Q1_train, Q2_train], y_train,
                         validation_split=0.1,
                         epochs=nepochs, batch_size=batch_size, shuffle=True,
                         class_weight={1: 0.46, 0: 1.32},
                         callbacks=[early_stopping, model_checkpoint], verbose=1)

        model.save_weights(xpfile + '.h5', overwrite=True)

        plot_training(hist, xpfile)
        # model.load_weights(xpfile + '.h5')
        return model

    def BIRNN_ATTENTION_LAYERS(self, nb_words, word_embedding_matrix, optimizer='adadelta', loss='binary_crossentropy',
                               encode=True, hidden_unit=100, **kwargs):
        """
        Code is based on the paper "A decomposable attention model for natural language inference (2016)" proposed by Aparikh, Oscart, Dipanjand, Uszkoreit. See more detail on https://arxiv.org/abs/1606.01933
        Bahdanau, D., Chorowski, J., Serdyuk, D., Brakel, P., & Bengio, Y. (2016, March). End-to-end attention-based large vocabulary speech recognition. In Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE International Conference on (pp. 4945-4949). IEEE.

    "Reasoning about entailment with neural attention (2016)" proposed by Tim Rockta schel. See more detail on https://arxiv.org/abs/1509.06664

    "Neural Machine Translation by Jointly Learning to Align and Translate (2016)" proposed by Yoshua Bengio, Dzmitry Bahdanau, KyungHyun Cho. See more detail on https://arxiv.org/abs/1409.0473
        :return:
        """
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

        if loss == 'softmax':
            out = 2
        else:
            out = 1
        preds = AggregationLayer(hidden_unit, output_units=1)(comp_1, comp_2)

        return loss, optimizer, preds, sequence_1_input, sequence_2_input

    def BILSTM(self, nb_words, word_embedding_matrix, optimizer='adadelta', loss='binary_crossentropy', hidden_unit=100,
               trainable=False, **kwargs):

        embedding_layer, sequence_1_input, sequence_2_input = self.embedded_input(nb_words=nb_words,
                                                                                  word_embedding_matrix=word_embedding_matrix,
                                                                                  trainable=trainable)

        # lstm_layer1 = LSTM(hidden_unit, dropout=DROPOUT_RATE,  # bias_initializer='random_normal',recurrent_dropout=DROPOUT_RATE
        #                    )
        # lstm_layer1 = Bidirectional(GRU(hidden_unit,dropout=DROPOUT_RATE))
        lstm_layer1 = GRU(hidden_unit, dropout=DROPOUT_RATE)

        embedded_sequences_1 = embedding_layer(sequence_1_input)

        x1 = lstm_layer1(embedded_sequences_1)
        # x1 = lstm_layer(x1)
        # lstm_layer2 = LSTM(hidden_unit, dropout=DROPOUT_RATE,  # bias_initializer='random_normal',
        # recurrent_dropout=DROPOUT_RATE
        # )
        # lstm_layer2 = Bidirectional(GRU(hidden_unit, dropout=DROPOUT_RATE))
        lstm_layer2 = GRU(hidden_unit, dropout=DROPOUT_RATE)
        embedding_layer2 = self.embedded_input(nb_words=nb_words, word_embedding_matrix=word_embedding_matrix,
                                               trainable=trainable, noinput=True)

        embedded_sequences_2 = embedding_layer2(sequence_2_input)
        x2 = lstm_layer2(embedded_sequences_2)
        # x2 = lstm_layer(x2)
        m = merge([x1, x2], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])
        # m = concatenate([x1, x2])
        # m = BatchNormalization()(m)

        # m = Dense(hidden_unit * 2, activation='relu')(m)
        m = Dense(hidden_unit * 2, activation='relu')(m)
        # m = BatchNormalization()(m)
        # m = Dense(int(hidden_unit), activation='sigmoid')(m)
        # m = BatchNormalization()(m)

        preds = Dense(1, activation='sigmoid', )(m)

        return loss, optimizer, preds, sequence_1_input, sequence_2_input

    def BICONV1DNN(self, nb_words, word_embedding_matrix, optimizer='adadelta', loss='binary_crossentropy',
                   hidden_unit=100,
                   trainable=False, **kwargs):
        num_filters = 16
        filter_sizes = [3, 4, 5]
        embedding_layer, sequence_1_input, sequence_2_input = self.embedded_input(nb_words=nb_words,
                                                                                  word_embedding_matrix=word_embedding_matrix,
                                                                                  trainable=trainable)

        embedded_sequences_1 = embedding_layer(sequence_1_input)

        conv_blocks = []
        for ks in filter_sizes:
            conv = Conv1D(filters=num_filters*2,
                          kernel_size=ks,
                          padding='valid',activation='relu',

                          strides=1)(embedded_sequences_1)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Conv1D(filters=num_filters,
                          kernel_size=ks,
                          padding='valid', activation='relu',
                          strides=1)(conv)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = SpatialDropout1D(DROPOUT_RATE / 2)(conv)
            conv = Conv1D(filters=num_filters,
                          kernel_size=ks,
                          padding='causal', activation='relu',
                          strides=1)(conv)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        x1 = merge(conv_blocks, mode='concat') if len(conv_blocks) > 1 else conv_blocks[0]


        embedded_sequences_2 = embedding_layer(sequence_2_input)
        conv_blocks = []
        for ks in filter_sizes:
            conv2 = Conv1D(filters=num_filters*2,
                           kernel_size=ks,
                           padding='valid',
                           activation='relu',
                           strides=1)(embedded_sequences_2)

            conv2 = MaxPooling1D(pool_size=2)(conv2)
            conv2 = Conv1D(filters=num_filters ,
                           kernel_size=ks,
                           padding='valid',
                           activation='relu',
                           strides=1)(conv2)
            # conv2 = LeakyReLU()(conv2)
            conv2 = MaxPooling1D(pool_size=2)(conv2)
            conv2 = SpatialDropout1D(DROPOUT_RATE / 2)(conv2)
            conv2 = Conv1D(filters=num_filters,
                           kernel_size=ks,
                           padding='causal',
                           activation='relu',
                           strides=1)(conv2)
        # conv2 = LeakyReLU()(conv2)
            conv2 = MaxPooling1D(pool_size=2)(conv2)
            conv2 = Flatten()(conv2)
            conv_blocks.append(conv2)
        x2 = merge(conv_blocks, mode='concat') if len(conv_blocks) > 1 else conv_blocks[0]

        # convs2 = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        m = merge([x1, x2], mode=lambda x: tf.abs(x[0] - x[1]), output_shape=lambda x: x[0])
        # m = merge([x1, x2], mode='dot')
        # m = concatenate([x1,x2 ])
        m = Dropout(DROPOUT_RATE)(m)
        m = BatchNormalization()(m)
        m = Dense(hidden_unit * 2, activation='relu', kernel_initializer='he_normal' )(m)
        m = Dropout(DROPOUT_RATE/2)(m)
        m = Dense(hidden_unit , activation='relu', kernel_initializer='he_normal')(m)
        preds = Dense(1, activation='sigmoid', kernel_initializer='zero')(m)

        return loss, optimizer, preds, sequence_1_input, sequence_2_input

    def BICONVLSTM(self, nb_words, word_embedding_matrix, optimizer='adadelta', loss='binary_crossentropy',
                   hidden_unit=100,
                   trainable=False, **kwargs):
        num_filters = 64
        filter_sizes = [3, 4]
        lstm_layer = LSTM((hidden_unit + hidden_unit / 2), dropout=DROPOUT_RATE, bias_initializer='random_normal', )
        embedding_layer, sequence_1_input, sequence_2_input = self.embedded_input(nb_words=nb_words,
                                                                                  word_embedding_matrix=word_embedding_matrix,
                                                                                  trainable=trainable)

        embedded_sequences_1 = embedding_layer(sequence_1_input)

        conv_blocks = []
        for ks in filter_sizes:
            conv = Conv1D(filters=num_filters,
                          kernel_size=ks,
                          padding='valid',
                          activation="relu",
                          strides=1)(embedded_sequences_1)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = SpatialDropout1D(DROPOUT_RATE)(conv)
            # conv = Flatten()(conv)
            # conv = TimeDistributed(conv)
            conv = GRU(hidden_unit, dropout=DROPOUT_RATE)(conv)
            conv_blocks.append(conv)
        x1 = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        embedding_layer = self.embedded_input(nb_words=nb_words,
                                              word_embedding_matrix=word_embedding_matrix,
                                              trainable=False, noinput=True)

        embedded_sequences_2 = embedding_layer(sequence_2_input)
        conv_blocks = []
        for ks in filter_sizes:
            conv2 = Conv1D(filters=num_filters,
                           kernel_size=ks,
                           padding='valid',
                           activation="relu",
                           strides=1)(embedded_sequences_2)
            conv2 = MaxPooling1D(pool_size=2)(conv2)
            conv2 = SpatialDropout1D(DROPOUT_RATE)(conv2)
            # conv2 = Flatten()(conv2)
            # conv2 = TimeDistributed(conv)
            conv2 = GRU(hidden_unit, dropout=DROPOUT_RATE)(conv2)
            conv_blocks.append(conv2)
        # x2 = merge(conv_blocks, mode='concat') if len(conv_blocks) > 1 else conv_blocks[0]
        x2 = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        m1 = merge([x1, x2], mode=lambda x: tf.abs(x[0] - x[1]), output_shape=lambda x: x[0])
        m2 = merge([x1, x2], mode='mul')
        m = concatenate([m1, m2])
        m = Dropout(DROPOUT_RATE)(m)
        # m = BatchNormalization()(m)

        m = Dense(hidden_unit * 2, activation='relu')(m)

        # m = BatchNormalization()(m)
        preds = Dense(1, activation='sigmoid')(m)

        return loss, optimizer, preds, sequence_1_input, sequence_2_input

    def BICONV_MERGELSTM(self, nb_words, word_embedding_matrix, optimizer='adadelta', loss='binary_crossentropy',
                         hidden_unit=100,
                         trainable=False, **kwargs):
        num_filters = 128
        filter_sizes = [3]
        embedding_layer, sequence_1_input, sequence_2_input = self.embedded_input(nb_words=nb_words,
                                                                                  word_embedding_matrix=word_embedding_matrix,
                                                                                  trainable=trainable)

        embedded_sequences_1 = embedding_layer(sequence_1_input)

        conv_blocks = []
        for ks in filter_sizes:
            conv = Conv1D(filters=num_filters,
                          kernel_size=ks,
                          padding='valid',
                          activation="relu",
                          strides=1)(embedded_sequences_1)
            conv = MaxPooling1D(pool_size=2)(conv)
            # conv = SpatialDropout1D(DROPOUT_RATE)(conv)
            conv_blocks.append(conv)
        x1 = concatenate(conv_blocks, axis=1) if len(conv_blocks) > 1 else conv_blocks[0]

        # embedding_layer = self.embedded_input(nb_words=nb_words,
        #                                       word_embedding_matrix=word_embedding_matrix,
        #                                       trainable=False, noinput=True)

        embedded_sequences_2 = embedding_layer(sequence_2_input)
        conv_blocks = []
        for ks in filter_sizes:
            conv2 = Conv1D(filters=num_filters,
                           kernel_size=ks,
                           padding='valid',
                           activation="relu",
                           strides=1)(embedded_sequences_2)
            conv2 = MaxPooling1D(pool_size=2)(conv2)
            # conv2 = SpatialDropout1D(DROPOUT_RATE)(conv2)
            conv_blocks.append(conv2)
        x2 = concatenate(conv_blocks, axis=1) if len(conv_blocks) > 1 else conv_blocks[0]

        # m1 = merge([x1, x2], mode=lambda x: tf.abs(x[0] - x[1]), output_shape=lambda x: x[0])
        # m2 = merge([x1, x2], mode='mul')
        m = concatenate([x1, x2])
        # m=m1
        # m = GRU(hidden_unit, activation='relu', dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE,return_sequences=True)(m)
        m = GRU(hidden_unit * 2, activation='relu', dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE / 2)(m)

        m = Dense(hidden_unit, kernel_initializer='he_normal', activation='relu')(m)

        preds = Dense(1, activation='sigmoid')(m)

        return loss, optimizer, preds, sequence_1_input, sequence_2_input

    def embedded_input(self, nb_words, word_embedding_matrix, trainable, noinput=False):
        embedding_layer = Embedding(nb_words + 1,
                                    EMBEDDING_DIM,
                                    weights=[word_embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=trainable)
        if noinput:
            return embedding_layer
        sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        return embedding_layer, sequence_1_input, sequence_2_input


def createBaseNetworkSmaller(inputLength, inputDim):
    baseNetwork = Sequential()
    baseNetwork.add(Conv1D(256, 7, strides=1, activation='relu', input_shape=(inputLength, inputDim),
                           padding='valid', kernel_initializer=RandomNormal(mean=0.0, stddev=0.05),
                           bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Conv1D(256, 3, strides=1, padding='valid', activation='relu', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05)))
    baseNetwork.add(MaxPooling1D(pool_size=3, strides=3))
    baseNetwork.add(Flatten())
    baseNetwork.add(Dense(32, activation='relu'))
    baseNetwork.add(Dropout(0.5))
    return baseNetwork


def reloadmodel(model, xpfile):
    """

    :param xpfile:
    :return:
    """
    print ('Load  already trained weights from the file %s' % xpfile)
    model.load_weights(xpfile + '.h5')
    return model


def run():
    batch_size, duplicate, encode, hidden_unit, name, nepochs, optimizer, trainable, ttype_token, ttype_w2v, validation, xpfile = configure()
    Q1_train, Q2_train, y_train, embedding_matrix, nb_words, tokenizer, validation_data = build_training_phase_data(
        validation=validation, duplicate=duplicate, ttype=ttype_w2v, tokentype=ttype_token,
        maxsequencelength=MAX_SEQUENCE_LENGTH)

    algo = DEEPNETWORKS(name=name)

    model = algo(nb_words=nb_words, word_embedding_matrix=embedding_matrix, optimizer=optimizer,
                 loss='binary_crossentropy', encode=encode, hidden_unit=hidden_unit, trainable=trainable)

    print model.summary()
    model = algo.train(model=model, Q1_train=Q1_train, Q2_train=Q2_train, y_train=y_train, batch_size=batch_size,
                       xpfile=xpfile, nepochs=nepochs)
    # model.load_weights(model_weights_file)
    del Q1_train, Q2_train, y_train
    if validation_data is not None:
        algo.test_method(model, validation_data, xpfile)

    submission = build_validation_submission_data(tokenizer=tokenizer, embedding_matrix=embedding_matrix,
                                                  nb_words=nb_words, maxsequencelength=MAX_SEQUENCE_LENGTH)

    algo.submit_score(submission, model, xpfile)

    print "LOAD FINAL_TEST"


def configure():
    config = pd.Series()
    # name = 'biconv'
    name = 'bilstm'
    # name = 'conv1D'
    # name = 'biconv1dlstm'
    # name = 'biconv1d_mergelstm'
    config['name'] = name
    validation = True
    config['validation'] = validation
    v = 'VYES' if validation else 'VNO'
    duplicate = False
    config['duplicate'] = duplicate
    d = 'DYES' if validation else 'DNO'
    ttype_token = 'only_train'
    config['ttype_token'] = ttype_token
    t = 'TOT' if ttype_token == 'only_train' else 'TALLT'
    # ttype_w2v = "glove"
    ttype_w2v = "twitter"
    config['ttype_w2v'] = ttype_w2v
    w = 'WGOO' if ttype_w2v == 'google'  else 'WGLO' if ttype_w2v == 'glove' else 'WTIW'
    if ttype_w2v == 'twitter':
        global EMBEDDING_DIM
        EMBEDDING_DIM = 50
    optimizer = 'adam'
    config['optimizer'] = optimizer
    encode = False
    config['encode'] = encode
    trainable = False
    config['trainable'] = trainable
    z = 'ET' if trainable  else 'E'
    info = 'with GRU'
    infoenc = ''
    if encode and name == 'birnnattention':
        infoenc = '_enc'
    path = bool_dir(os.path.join(XP_FOLDER, name + infoenc))
    xpname = '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(get_time(), t, w, v, d, trainable, info)
    xpfile = os.path.join(path, xpname)
    config['xpfile'] = xpfile
    print('The model can be found here at %s.h5' % xpfile)
    batch_size = 512
    config['batch_size'] = batch_size
    nepochs = 100
    config['nepochs'] = nepochs
    hidden_unit = 1024
    config['hidden_unit'] = hidden_unit
    conf = pd.DataFrame().append(config.T, ignore_index=True)
    conf.to_csv(xpfile + '_configuration.csv', index=False)
    return batch_size, duplicate, encode, hidden_unit, name, nepochs, optimizer, trainable, ttype_token, ttype_w2v, validation, xpfile


# def reload_submit():

if __name__ == '__main__':
    run()
