import os
import gc
from nltk.corpus import stopwords
from time import gmtime, strftime
import seaborn.apionly as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
def get_time():
    return strftime("%mM_%dD__%Hh_%Mmin", gmtime())
try:
    import cPickle as pickle
except:
    import pickle


def train_test_split(X, y, test_size=0.1):
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=test_size)
    train, test = [(a, b) for a, b in sss][0]
    return X[train], X[test], y[train], y[test]



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
    plt.savefig(model_weights_file + '_plot_training.png', bbox_inches='tight')
def bool_dir(path):
    """
    Check if directories exist else it create all the dirs

    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def cut_test():
    import pandas as pd
    import numpy as np

    df = pd.read_csv('data/test.csv')
    x = np.array_split(df,20)
    i=1
    for d in x:
        d.to_csv('data/test_part{0}.csv'.format(i),index=False)
        i+=1


    # Compute the correlation matrix
    # corr = Xtrain[col].corr()
    #


    # Generate a mask for the upper triangle
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    # hh=sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
    #             square=True,# xticklabels=5, yticklabels=5,
    #             xticklabels=xticks,yticklabels=yticks,
    # linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    # for item in hh.get_xticklabels():
    #     item.set_rotation(90)
    # for item in hh.get_yticklabels():
    #     item.set_rotation(360)

    # sns.plt.savefig("correlationplot1.png",)