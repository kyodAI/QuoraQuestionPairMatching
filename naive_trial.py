from tools import bool_dir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

def createDTM(messages):
    vect = TfidfVectorizer()
    vect.fit(messages) # create DTM
    dtm = vect.transform(messages)

    # create pandas dataframe of DTM
    return dtm,vect


def load_train():
    df = pd.read_csv("data/train.csv")
    df.replace(np.nan, '', inplace=True)
    return df


def load_test():
    df = pd.read_csv("data/test.csv")
    df.replace(np.nan, '', inplace=True)

    return df

def get_questions_pair_separatelists(df,istestdf=False):
    if istestdf:
        dfq1 = df[[ 'question1']].copy()
        dfq2 = df[['question2']].copy()
    else:
        dfq1 = df[['qid1', 'question1']].copy()
        dfq2 = df[['qid2', 'question2']].copy()
    return dfq1,dfq2

def preprocess_data_train(df):
    dfq1, dfq2 = get_questions_pair_separatelists(df)

    concatdf = pd.DataFrame()
    dtpm = pd.DataFrame()
    dtpm['qid'] = dfq1['qid1']
    dtpm['question'] = dfq1['question1']
    concatdf = concatdf.append(dtpm, ignore_index=True)
    dtpm = pd.DataFrame()
    dtpm['qid'] = dfq2['qid2']
    dtpm['question'] = dfq2['question2']
    concatdf = concatdf.append(dtpm, ignore_index=True)
    concatdf.drop_duplicates('qid', inplace=True)
    concatdf.dropna(inplace=True)
    questions = list(concatdf['question'])
    transformed_data, feature_extractor = createDTM(questions)

    question1, question2 = questions_feature_extraction(dfq1, dfq2, feature_extractor)

    return feature_extractor,question1,question2


def questions_feature_extraction(dfq1, dfq2, feature_extractor):
    question1 = list(dfq1['question1'])
    question1 = feature_extractor.transform(question1)
    question2 = list(dfq2['question2'])
    question2 = feature_extractor.transform(question2)
    return question1, question2


def merge(qmatrix1,qmatrix2,ttype="substract"):
    if ttype=='substract':
        return qmatrix1-qmatrix2
    if ttype == 'multiply':
        return np.multiply(qmatrix1,qmatrix2.T)



def train(mergettype='substract'):
    print('Load Train Data')
    df = load_train()
    print('Extract Features')
    feature_extractor, question1,question2 = preprocess_data_train(df)
    print('Fit the Random Forest Classifier')
    Xtrain = merge(question1,question2,ttype=mergettype)
    ytrain = np.array(df['is_duplicate'])
    estimator = RandomForestClassifier(n_jobs=4,n_estimators=25)
    estimator.fit(Xtrain, ytrain)
    return estimator,feature_extractor

def test(estimator,feature_extractor,mergettype='substract'):
    """

    :param estimator:
    :return:
    """
    print('Load Test Data')
    df = load_test()
    dfq1, dfq2 = get_questions_pair_separatelists(df,istestdf=True)
    print('Extract The Features of Questions in the Test Data ')
    question1, question2 = questions_feature_extraction(dfq1, dfq2, feature_extractor)
    print('Transform the Test Data to  cope with an Information Retrieval Problem')
    Xtest = merge(question1, question2,ttype=mergettype)

    ypred = estimator.predict(Xtest)
    result = pd.DataFrame()
    result['test_id'] = df['test_id']
    result['is_duplicate'] = ypred
    return result



if __name__ == '__main__':
    mergettype = 'multiply'
    estimator,feature_extractor = train(mergettype=mergettype)
    result = test(estimator=estimator,feature_extractor=feature_extractor,mergettype=mergettype)
    result.to_csv(bool_dir('results')+'/submission_naive_method_m{0}.csv'.format(mergettype),index=False)
