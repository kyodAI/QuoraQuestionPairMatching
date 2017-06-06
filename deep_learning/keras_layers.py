import keras.backend as K
from keras.layers import Lambda, Activation, Dropout, Embedding, SpatialDropout1D, Dense, merge
from keras.layers import TimeDistributed  # This applies the model to every timestep in the input sequences
from keras.layers import Bidirectional, GRU, LSTM
import numpy as np
from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras.callbacks import Callback

from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate

class EmbeddingLayer(object):

    def __init__(self, vocab_size, embedding_size, max_length, output_units, init_weights=None, nr_tune=1000, dropout=0.0):
        self.output_units = output_units
        self.max_length = max_length
        self.dropout = dropout

        self.embed = Embedding(
            vocab_size,
            embedding_size,
            input_length=max_length,
            weights=[init_weights],
            name='embedding',
            trainable=False
        )

        self.tune = Embedding(
            nr_tune,
            output_units,
            input_length=max_length,
            weights=None,
            name='tune',
            trainable=True,
        )

        self.mod_ids = Lambda(lambda sent: sent % (nr_tune - 1) + 1, output_shape=(self.max_length,))

        self.project = TimeDistributed(Dense(output_units, use_bias=False, name='project'))

    def __call__(self, sentence):

        def get_output_shape(shapes):
            print(shapes)

            return shapes[0]

        mod_sent = self.mod_ids(sentence)

        # SpatialDropout1D drops entire 1D feature maps instead of individual elements
        tuning = SpatialDropout1D(self.dropout)(self.tune(mod_sent))

        # tuning = merge([tuning, mod_sent],
        #    mode=lambda AB: AB[0] * (K.clip(K.cast(AB[1], 'float32'), 0, 1)),
        #    output_shape=(self.max_length, self.output_units))
        pretrained = self.project(self.embed(sentence))
        vectors = merge([pretrained, tuning], mode='sum')

        return vectors


class BiRNN_EncodingLayer(object):

    def __init__(self, max_length, hidden_units, dropout=0.0):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout), input_shape=(max_length, hidden_units)))  # return_sequences: return the last output in the output sequence, or the full sequence.
        self.model.add(TimeDistributed(Dense(hidden_units, activation='relu', kernel_initializer='he_normal')))
        self.model.add(TimeDistributed(Dropout(0.2)))

    def __call__(self, embedded_words):
        return self.model(embedded_words)


class AttentionLayer(object):

    def __init__(self, max_length, hidden_units, dropout=0.0, l2_weight_decay=0.0, activation='relu'):

        """
        F function => attention = transpose of F(a) * F(b)
        """
        self.max_length = max_length
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(hidden_units,)))
        self.model.add(Dense(hidden_units, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_weight_decay), name='attend1'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_units, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_weight_decay), activation='relu', name='attend2'))
        self.model = TimeDistributed(self.model)  # Apply attention for each timestep

    def __call__(self, sent1, sent2):
        def _outer(AB):
            """
            Calculate unnormalized attention weights
            """
            energy = K.batch_dot(x=AB[1], y=K.permute_dimensions(AB[0], pattern=(0, 2, 1)))
            return K.permute_dimensions(energy, (0, 2, 1))

        return merge(inputs=[self.model(sent1), self.model(sent2)],
                     mode=_outer,
                     output_shape=(self.max_length, self.max_length))


class SoftAlignmentLayer(object):

    def __init__(self, max_length, hidden_units):
        self.max_length = max_length
        self.hidden_units = hidden_units

    def __call__(self, sentence, attention, transpose=False):

        def _normalize_attention(attention_and_sent):
            attention = attention_and_sent[0]   # attention matrix   shape=(?, max_length, max_length)
            sentence = attention_and_sent[1]    # sentence that wants to be aligned   shape=(?, max_length, embedding_size)

            if transpose:
                attention = K.permute_dimensions(attention, (0, 2, 1))

            # 3D softmax - calculate the subphrase in the sentence through attention
            exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
            summation = K.sum(exp, axis=-1, keepdims=True)
            weights = exp / summation  # (512, 512)
            sub_phrase_in_sentence = K.batch_dot(weights, sentence)

            return sub_phrase_in_sentence

        return merge([attention, sentence],
                     mode=_normalize_attention,
                     output_shape=(self.max_length, self.hidden_units))


class ComparisonLayer(object):
    """
    Separately compare the aligned phrases using a function "G"
    """

    def __init__(self, words, hidden_units, l2_weight_decay=0.0, dropout=0.0):
        self.words = words

        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(hidden_units * 2,)))
        self.model.add(Dense(hidden_units, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_weight_decay), name='compare1'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_units, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_weight_decay), name='compare2'))
        self.model.add(Activation('relu'))
        self.model = TimeDistributed(self.model)  # Apply comparison for each timestep

    def __call__(self, sent, align, **kwargs):
        result = self.model(merge([sent, align], mode='concat'))  # Shape: (batch, max_length, 2 * hidden_units)
        avged = GlobalAveragePooling1D()(result)
        maxed = GlobalMaxPooling1D()(result)
        merged = merge([avged, maxed], mode='sum')
        result = BatchNormalization()(merged)

        return result


class AggregationLayer(object):
    """
    Concatenate two sets of comparison vectors and aggregate over each set by summation
    y = H([v1, v2])
    """

    def __init__(self, hidden_units, output_units, dropout=0.0, l2_weight_decay=0.0):
        # Define H function
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(hidden_units * 2,)))
        self.model.add(Dense(hidden_units, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_weight_decay)))
        self.model.add(Activation('relu'))

        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_units, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_weight_decay)))
        self.model.add(Activation('relu'))

        self.model.add(Dense(output_units, activation='sigmoid', kernel_initializer='zero', kernel_regularizer=regularizers.l2(l2_weight_decay)))

    def __call__(self, feats1, feats2):
        return self.model(merge([feats1, feats2], mode='concat'))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class AdaptiveLearningRateScheduler(Callback):
    """Learning rate scheduler that increases or decreases LR based on a recent sample of validation results"""

    def __init__(self, initial_learning_rate, monitor="val_loss", scale=2., window=5):
        super(AdaptiveLearningRateScheduler, self).__init__()
        self.monitor = monitor
        self.initial_lr = initial_learning_rate
        self.scale = float(scale)
        self.window = window

        self.metric_ = []

    def on_epoch_begin(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), 'Optimizer must have a "lr" attribute.'

        lr = self._get_learning_rate()

        if lr:
            K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs={}):
        metric = logs[self.monitor]
        self.metric_.append(metric)

    def _get_learning_rate(self):
        if len(self.metric_) < self.window * 2:
            return self.initial_lr

        data = np.asarray(self.metric_)

        baseline = data[:-self.window].min()
        diffs = baseline - data[-self.window:]

        # assume error, lower is better
        percent_epochs_improved = sigmoid((diffs / baseline) / 0.02).mean()

        if percent_epochs_improved > 0.75:
            return self._scale_learning_rate(self.scale)
        elif percent_epochs_improved < 0.5:
            return self._scale_learning_rate(1. / self.scale)

        return None

    def _scale_learning_rate(self, scale):
        return K.get_value(self.model.optimizer.lr) * scale