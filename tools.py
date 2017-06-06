import os
import gc
from nltk.corpus import stopwords
STOPS = set(stopwords.words('english'))

try:
    import cPickle as pickle
except:
    import pickle


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