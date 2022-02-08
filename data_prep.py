import os
import pandas as pd
import re
import numpy as np
from scipy.sparse import coo_matrix
from gensim.utils import simple_preprocess
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import contractions


def import_chi_review():
    file_dir = 'D://Python Projects/Fake_Review_Detection/Data/Chicago_Hotel_Review'
    neg_dec_dir = file_dir + '/negative_polarity/deceptive_from_MTurk'
    neg_tru_dir = file_dir + '/negative_polarity/truthful_from_Web'
    pos_dec_dir = file_dir + '/positive_polarity/deceptive_from_MTurk'
    pos_tru_dir = file_dir + '/positive_polarity/truthful_from_TripAdvisor'

    neg_dec_list = [file_name for file_name in os.listdir(f'{neg_dec_dir}') if file_name.endswith('.txt')]
    neg_tru_list = [file_name for file_name in os.listdir(f'{neg_tru_dir}') if file_name.endswith('.txt')]
    pos_dec_list = [file_name for file_name in os.listdir(f'{pos_dec_dir}') if file_name.endswith('.txt')]
    pos_tru_list = [file_name for file_name in os.listdir(f'{pos_tru_dir}') if file_name.endswith('.txt')]

    neg_dec_txt = []
    neg_tru_txt = []
    pos_dec_txt = []
    pos_tru_txt = []
    for neg_dec_name, neg_tru_name, pos_dec_name, pos_tru_name in zip(neg_dec_list, neg_tru_list, pos_dec_list,
                                                                      pos_tru_list):
        neg_dec_file = open(neg_dec_dir + '/' + neg_dec_name, 'r')
        neg_tru_file = open(neg_tru_dir + '/' + neg_tru_name, 'r')
        pos_dec_file = open(pos_dec_dir + '/' + pos_dec_name, 'r')
        pos_tru_file = open(pos_tru_dir + '/' + pos_tru_name, 'r')
        neg_dec_txt.append(neg_dec_file.read().strip())
        neg_tru_txt.append(neg_tru_file.read().strip())
        pos_dec_txt.append(pos_dec_file.read().strip())
        pos_tru_txt.append(pos_tru_file.read().strip())
        neg_dec_file.close()
        neg_tru_file.close()
        pos_dec_file.close()
        pos_tru_file.close()

    rating = [1] * 800 + [5] * 800
    all_text = neg_dec_txt + neg_tru_txt + pos_dec_txt + pos_tru_txt
    label = [1] * 400 + [-1] * 400 + [1] * 400 + [-1] * 400
    df = pd.DataFrame({'Label': label, 'Rating': rating, 'Ori_Review': all_text})
    return df

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(words):
    result = []
    lm = WordNetLemmatizer()
    for word, pos in nltk.pos_tag(words):
        lm_pos = get_wordnet_pos(pos)
        result.append(lm.lemmatize(word, lm_pos))
    return result
    
    
def remove_stopwords(sentence):
    stop_words = stopwords.words('english')
    stop_words.extend(['chicago', 'hotel', 'would', 'could', 'should', 'might', 'room', 'stay'])
    result = [word for word in sentence if word not in stop_words]
    return result


# Remove Punctuation and turn all letter to lowercase
def preprocess_ngram(df):
    df_pp = df.copy()
    clean_review = df_pp['Ori_Review'].copy()
    clean_review = clean_review.apply(lambda x: x.lower())
    contraction_dict = contractions.gen_contractions()
    for con, full in contraction_dict.items():
        clean_review = clean_review.apply(lambda x: re.sub(con, full, x))
    clean_review = clean_review.apply(lambda x: re.sub('[,/.!?]', '', x))
    words = clean_review.apply(lambda x: simple_preprocess(x, deacc=True, min_len=1))
    words = words.apply(lambda x: lemmatize(x))
    df_pp['Clean_Review'] = words.apply(lambda x: ' '.join(x))
    words = words.apply(lambda x: remove_stopwords(x))
    df_pp['PP_Review'] = words.apply(lambda x: ' '.join(x))
    df_pp['Word_List'] = words
    df_pp = df_pp.reset_index(drop=True)
    return df_pp


def preprocess_ling_feature(df):
    df_pp = df.copy()
    clean_review = df_pp['Ori_Review'].copy()
    clean_review = clean_review.apply(lambda x: x.lower())
    contraction_dict = contractions.gen_contractions()
    for con, full in contraction_dict.items():
        clean_review = clean_review.apply(lambda x: re.sub(con, full, x))
    clean_review = clean_review.apply(lambda x: re.sub('[,/.!?]', '', x))
    words = clean_review.apply(lambda x: x.split())
    words = words.apply(lambda x: lemmatize(x))
    df_pp['Word_List_all'] = words
    return df_pp


def df2matrix(df, word2ind):
    rows = []
    cols = []
    for r, document in enumerate(df['Ngram'].tolist()):
        for word in document:
            rows.append(r)
            cols.append(word2ind[word])
    vals = np.array([1] * len(rows))
    X = coo_matrix((vals, [rows, cols]), shape=(max(rows) + 1, len(word2ind))).toarray()
    X = pd.DataFrame(X, columns=list(word2ind.keys()))
    X['Rating'] = df['Rating'].copy()
    ling_fea = ['num_word', 'num_coreword', 'num_stopword', 'num_char', 'char_per_word', 'num_first_sing']
    X[ling_fea] = df[ling_fea].copy()
    y = df['Label'].copy()
    return X, y