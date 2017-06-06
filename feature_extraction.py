import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import re
from fuzzywuzzy import fuzz
import gensim
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis


pal = sns.color_palette()

STOPS = set(stopwords.words('english'))


def load_dataset(ttype='train', clean=False):
    if ttype not in ['train', 'test',
                     'test_part1',
                     'test_part2',
                     'test_part3',
                     'test_part4',
                     'test_part5',
                     'test_part6',
                     'test_part7',
                     'test_part8',
                     'test_part9',
                     'test_part10',
                     'test_part11',
                     'test_part12',
                     'test_part13',
                     'test_part14',
                     'test_part15',
                     'test_part16',
                     'test_part17',
                     'test_part18',
                     'test_part19',
                     'test_part20', ]:
        raise AssertionError("need  to use train or test keyword")
    cleanfile = '_clean' if clean else ''

    df = pd.read_csv('/media/nacim/a13ff970-2886-43fb-b752-f1a813a76a12/data/{0}{1}.csv'.format(ttype, cleanfile)).replace(np.nan, '')
    return df


class Extractor():
    stem = False
    lemmatize = False
    dropstop = False
    dict_features = {}
    model = None
    modelisnormed = False

    def __init__(self, stem=False, lemmatize=False, dropstop=True):
        self.__class__.stem = stem
        self.__class__.lemmatize = lemmatize
        self.__class__.dropstop = dropstop

    @classmethod
    def get_wordnet_pos(cls, treebank_tag):
        """
        Lemmatization
        :param treebank_tag:
        :return:
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    @classmethod
    def lemmatize_sentence(cls, sentence):
        """
        Lemmatization of the sentence
        :param sentence:
        :return:
        """
        res = []
        lemmatizer = WordNetLemmatizer()
        for word, pos in pos_tag(word_tokenize(sentence)):
            wordnet_pos = cls.get_wordnet_pos(pos) or wordnet.NOUN
            res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
        return res

    #
    @classmethod
    def stem_words(cls, sentence):
        """
        Stem words
        :param sentence:
        :return:
        """
        porter_stemmer = PorterStemmer()

        # If sentence is a string, transfer to array
        if type(sentence) == str:
            sentence = sentence.split()

        st_lst = []
        for w in sentence:
            w = porter_stemmer.stem(w)
            st_lst.append(w)

        return st_lst

    @classmethod
    def remove_stopwords(cls, sentence):
        # Input a str or list type sentence, remove the stopwords
        # Return a word list

        # If sentence is a string, transfer to array
        if type(sentence) == str:
            sentence = sentence.split()

            # return type is list
            #     ns_lst = []
            #     for w in sentence:
            #         if w not in stops:
            #             ns_lst.append(w)

        # return type is dict
        ns_words = {}
        for word in sentence:
            if word not in STOPS:
                ns_words[word] = 1

        return ns_words

    # Word matching
    @classmethod
    def remove_contraction(cls, text):

        text = " ".join(text)
        # Clean the text
        # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i m", "i am", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'m", " am ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"60k", " 60000 ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        # text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        # text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"quikly", "quickly", text)
        text = re.sub(r"usa", "America", text)
        text = re.sub(r"canada", "Canada", text)
        text = re.sub(r"japan", "Japan", text)
        text = re.sub(r"germany", "Germany", text)
        text = re.sub(r"burma", "Burma", text)
        text = re.sub(r"rohingya", "Rohingya", text)
        text = re.sub(r"zealand", "Zealand", text)
        text = re.sub(r"cambodia", "Cambodia", text)
        text = re.sub(r"zealand", "Zealand", text)
        text = re.sub(r"norway", "Norway", text)
        text = re.sub(r" uk ", " England ", text)
        text = re.sub(r"india", "India", text)
        text = re.sub(r"pakistan", "Pakistan", text)
        text = re.sub(r"britain", "Britain", text)
        text = re.sub(r"switzerland", "Switzerland", text)
        text = re.sub(r"china", "China", text)
        text = re.sub(r"chinese", "Chinese", text)
        text = re.sub(r"imrovement", "improvement", text)
        text = re.sub(r"intially", "initially", text)
        text = re.sub(r"quora", "Quora", text)
        text = re.sub(r" dms ", "direct messages ", text)
        text = re.sub(r"demonitization", "demonetization", text)
        text = re.sub(r"actived", "active", text)
        text = re.sub(r"kms", " kilometers ", text)
        text = re.sub(r" cs ", " computer science ", text)
        text = re.sub(r" upvotes ", " up votes ", text)
        text = re.sub(r" iphone ", " phone ", text)
        # text = re.sub(r"\0rs ", " rs ", text)
        text = re.sub(r"calender", "calendar", text)
        text = re.sub(r"ios", "operating system", text)
        text = re.sub(r"gps", "GPS", text)
        text = re.sub(r"gst", "GST", text)
        text = re.sub(r"programing", "programming", text)
        text = re.sub(r"bestfriend", "best friend", text)
        text = re.sub(r"dna", "DNA", text)
        text = re.sub(r"iii", "3", text)
        text = re.sub(r"california", "California", text)
        text = re.sub(r"texas", "Texas", text)
        text = re.sub(r"tennessee", "Tennessee", text)
        text = re.sub(r"the us", "America", text)
        text = re.sub(r"trump", "Trump", text)
        return str(' '.join(text))

    @classmethod
    def word_match_share(cls, row):
        # Input a row of question pairs
        # Return (comm_word_cnt_q1 + comm_word_cnt_q2)/(word_cnt_q1 + word_cnt_q2)

        q1 = str(row['question1']).lower()
        q2 = str(row['question2']).lower()

        # Lemmatize
        if cls.lemmatize:
            q1 = cls.lemmatize_sentence(q1)
            # print q1
            # exit()
            q2 = cls.lemmatize_sentence(q2)

        # Remove stopwords
        if cls.dropstop:
            q1_words = cls.remove_stopwords(q1)
            q2_words = cls.remove_stopwords(q2)

        # Stemming
        if cls.stem:
            q1_words = cls.stem_words(q1_words)
            q2_words = cls.stem_words(q2_words)

        if len(q1_words) == 0 or len(q2_words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        shared_words_in_q1 = [w for w in q1_words if w in q2_words]

        shared_words_in_q2 = [w for w in q2_words if w in q1_words]
        R = float(len(shared_words_in_q1) + len(shared_words_in_q2)) / float(len(q1_words) + len(q2_words))
        return R

    @classmethod
    def tfidf_word_match_share(cls, row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in STOPS:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in STOPS:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        # shared_weights: q1 shared words' weight + q2 shared words' weight
        # total_weights: q1 all words' weight + q2 all words' weight
        shared_weights = np.array([cls.weights.get(w, 0) for w in q1words.keys() if w in q2words])
        + np.array([cls.weights.get(w, 0) for w in q2words.keys() if w in q1words])
        total_weights = np.sum([cls.weights.get(w, 0) for w in q1words]) + np.sum(
            [cls.weights.get(w, 0) for w in q2words])

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    @classmethod
    def get_weight(cls, count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1. / float(count + eps)

    @classmethod
    def set_weights(cls, df):
        qs = pd.Series(df['question1'].tolist() + df['question2'].tolist()).astype(str)
        eps = 5000
        words = (" ".join(qs)).lower().split()
        # counts is the dict format and count for the word in all questions
        counts = Counter(words)
        # weights is 1/(count+5000)

        cls.weights = {word: cls.get_weight(count) for word, count in counts.items()}

    @classmethod
    def load_model(cls, normed=False, init=False):

        if cls.model is not None:

            if (normed == False) and cls.modelisnormed == True:
                init = True
            if init:
                cls.model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
                    '/media/nacim/a13ff970-2886-43fb-b752-f1a813a76a12/data/GoogleNews-vectors-negative300.bin.gz', binary=True)
                if normed:
                    cls.model.init_sims(replace=True)
                    cls.modelisnormed = True
            else:
                if normed:
                    if not cls.modelisnormed:
                        cls.model.init_sims(replace=True)
                        cls.modelisnormed = True

        else:
            print('Model WordNet trained on googleNews is loading')
            cls.model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
                '/media/nacim/a13ff970-2886-43fb-b752-f1a813a76a12/data/GoogleNews-vectors-negative300.bin.gz', binary=True)
            if normed:
                cls.model.init_sims(replace=True)
                cls.model

    @classmethod
    def wmd(cls, s1, s2):
        s1 = str(s1).lower().split()
        s2 = str(s2).lower().split()
        s1 = [w for w in s1 if w not in STOPS]
        s2 = [w for w in s2 if w not in STOPS]
        return cls.model.wmdistance(s1, s2)

    @classmethod
    def norm_wmd(cls, s1, s2):
        s1 = str(s1).lower().split()
        s2 = str(s2).lower().split()
        s1 = [w for w in s1 if w not in STOPS]
        s2 = [w for w in s2 if w not in STOPS]
        return cls.model.wmdistance(s1, s2)

    @classmethod
    def resetmodel(cls):
        cls.model = None
        cls.modelisnormed = False
        gc.collect()

    @classmethod
    def features_fuzzy(cls, df):

        print("Start to calculate fuzzy features.")
        cls.dict_features['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])),
                                                    axis=1)
        print("1/7 QRatio finished.")

        cls.dict_features['fuzz_wratio'] = df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])),
                                                    axis=1)
        print("2/7 WRatio finished.")

        cls.dict_features['fuzz_partial_ratio'] = df.apply(
            lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
        print("3/7 Partial Ratio finished.")

        cls.dict_features['fuzz_partial_token_set_ratio'] = df.apply(
            lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        print("4/7 Partial Token Set Ratio finished.")

        cls.dict_features['fuzz_partial_token_sort_ratio'] = df.apply(
            lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        print("5/7 Partial Token Sort Ratio finished.")

        cls.dict_features['fuzz_token_set_ratio'] = df.apply(
            lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),
            axis=1)
        print("6/7 Token Set Ratio finished.")

        cls.dict_features['fuzz_token_sort_ratio'] = df.apply(
            lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])),
            axis=1)
        print("7/7 Token Sort Ratio finished.")

    @classmethod
    def features_distance_frommodel(cls, df):
        cls.load_model(normed=False)
        cls.dict_features['wmd_train'] = df.apply(lambda x: cls.wmd(x['question1'], x['question2']), axis=1)
        print("Word Mover Distance finished.")
        # cls.resetmodel()

        cls.load_model(normed=True, init=False)
        print("Normalize model built.")
        cls.dict_features['norm_wmd_train'] = df.apply(lambda x: cls.norm_wmd(x['question1'], x['question2']), axis=1)
        print("Normalize Word Mover Distince finished.")
        cls.resetmodel()

    @classmethod
    def sent2vec(cls, s):
        # words = str(s).lower().decode('utf-8')
        words = str(s).lower()
        try:

            words = word_tokenize(words)
        except:
            words = word_tokenize(unicode(words, errors='replace'))

        words = [w for w in words if not w in STOPS]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(cls.model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())

    @classmethod
    def get_questions_vector(cls, df):
        question1_vectors = np.zeros((df.shape[0], 300))
        error_count = 0

        for i, q in tqdm(enumerate(df.question1.values)):
            question1_vectors[i, :] = cls.sent2vec(q)

        question2_vectors = np.zeros((df.shape[0], 300))
        for i, q in tqdm(enumerate(df.question2.values)):
            question2_vectors[i, :] = cls.sent2vec(q)

        return question1_vectors, question2_vectors

    @classmethod
    def features_similarity(cls, df):
        cls.load_model(normed=True)
        question1_vectors, question2_vectors = cls.get_questions_vector(df)
        cls.resetmodel()

        cls.dict_features['cosine_distance'] = [cosine(x, y) for (x, y) in
                                                zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
        print("1/11 Cosine Distance finished.")
        cls.dict_features['cityblock_distance'] = [cityblock(x, y) for (x, y) in
                                                   zip(np.nan_to_num(question1_vectors),
                                                       np.nan_to_num(question2_vectors))]
        print("2/11 Cityblock Distance finished.")
        cls.dict_features['jaccard_distance'] = [jaccard(x, y) for (x, y) in
                                                 zip(np.nan_to_num(question1_vectors),
                                                     np.nan_to_num(question2_vectors))]
        print("3/11 Jaccard Distance finished.")
        cls.dict_features['canberra_distance'] = [canberra(x, y) for (x, y) in
                                                  zip(np.nan_to_num(question1_vectors),
                                                      np.nan_to_num(question2_vectors))]
        print("4/11 Canberra Distance finished.")
        cls.dict_features['euclidean_distance'] = [euclidean(x, y) for (x, y) in
                                                   zip(np.nan_to_num(question1_vectors),
                                                       np.nan_to_num(question2_vectors))]
        print("5/11 Euclidean Distance finished.")
        cls.dict_features['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in
                                                   zip(np.nan_to_num(question1_vectors),
                                                       np.nan_to_num(question2_vectors))]
        print("6/11 Minkowski Distance finished.")
        cls.dict_features['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in
                                                    zip(np.nan_to_num(question1_vectors),
                                                        np.nan_to_num(question2_vectors))]
        print("7/11 Braycurtis Distance finished.")
        cls.dict_features['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
        print("8/11 Skew Q1 Vec finished.")
        cls.dict_features['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
        print("9/11 Skew Q2 Vec finished.")
        cls.dict_features['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
        print("10/11 Kurtosis Q1 Vec finished.")
        cls.dict_features['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
        print("11/11 Kurtosis Q2 Vec finished.")
        return question1_vectors, question2_vectors

    def features_computation(cls, df):

        print ('Compute Simililarity between questions from their vector computed on Wordnet')
        question1, question2 = cls.features_similarity(df=df)

        print('Compute Distance between questions from Wordnet trained on Google')
        cls.features_distance_frommodel(df=df)

        print('Compute word match')
        cls.dict_features['word_match_train'] = df.apply(cls.word_match_share, axis=1, raw=True)
        cls.set_weights(df=df)
        print('Compute tfidf word match')
        cls.dict_features['tfidf_word_match_train'] = df.apply(cls.tfidf_word_match_share, axis=1, raw=True)

        print('Compute fuzzy  features')
        cls.features_fuzzy(df)

        df_extracted_features = pd.DataFrame()
        for key in cls.dict_features.keys():
            df_extracted_features[key] = cls.dict_features[key]

        df_extracted_features['len_q1'] = df.question1.apply(lambda x: len(str(x)))
        df_extracted_features['len_q2'] = df.question2.apply(lambda x: len(str(x)))
        df_extracted_features['diff_len'] = df_extracted_features.len_q1 - df_extracted_features.len_q2
        df_extracted_features['len_char_q1'] = df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        df_extracted_features['len_char_q2'] = df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        df_extracted_features['len_word_q1'] = df.question1.apply(lambda x: len(str(x).split()))
        df_extracted_features['len_word_q2'] = df.question2.apply(lambda x: len(str(x).split()))
        return df_extracted_features, question1, question2


def extract_features_train_dataset(ttype='train'):
    print ('Loading Dataset')
    df = load_dataset(ttype=ttype)
    print ('Start Extracting Features')
    print ('=========================')
    extractor = Extractor(lemmatize=False, dropstop=True)
    df_new_features, question1, question2 = extractor.features_computation(df=df)
    df_new_features.to_csv('/media/nacim/a13ff970-2886-43fb-b752-f1a813a76a12/data/{0}_extracted_features.csv'.format(ttype), index=False)
    tmp = pd.DataFrame(question1)
    tmp.to_csv('/media/nacim/a13ff970-2886-43fb-b752-f1a813a76a12/data/{0}_question1.csv'.format(ttype), index=False)

    tmp = pd.DataFrame(question2)
    tmp.to_csv('/media/nacim/a13ff970-2886-43fb-b752-f1a813a76a12/data/{0}_question2.csv'.format(ttype), index=False)


if __name__ == '__main__':

    for i in range(10, 21):
        ttype = 'test_part{0}'.format(i)

        extract_features_train_dataset(ttype=ttype)
    """
            .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
        .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
        .. Matt Kusner et al. "From Word Embeddings To Document Distances".
    """
