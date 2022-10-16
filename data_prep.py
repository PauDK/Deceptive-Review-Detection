import os
import pandas as pd
import re
import numpy as np
from gensim.utils import simple_preprocess
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import contractions
from scipy.sparse import coo_matrix


def import_chi_review():
    '''
    Import Chicago Hotel Review dataset
    Args:
        None
    Returns:
        df (pd.DataFrame): A dataframe with the following columns 
            - Label (int): -1 for truthful review, 1 for deceptive review 
            - Rating (int): 1 for negative review, 5 for positive reivew
            - Ori_Review (str): Original text review without any cleaning
    '''
    file_dir = 'Chicago_Hotel_Review'
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
    '''
    Get WordNet Part-of-Speech tagging
    Args:
        treebank_tag (str): string of detailed part-of-speech abbreviation
    Returns:
        Wordnet part-of-speech object grouping low-level part-of-speech to a higher level
    '''
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
    '''
    Lemmatize each token in words through WordNet's Lemmatizer.
    Ex. see, saw, seen -> see
    Args:
        words (list): list of tokens
    Returns:
        lemmatized_words (list): list of lemmatized token
    '''
    lemmatized_words = []
    lm = WordNetLemmatizer()
    for word, pos in nltk.pos_tag(words):
        lm_pos = get_wordnet_pos(pos)
        lemmatized_words.append(lm.lemmatize(word, lm_pos))
    return lemmatized_words
    
    
def remove_stopwords(words):
    '''
    Remove stopwords (ex. by, for, from, the, to) using NLTK's stopwords vocab.
    Args:
        words (list): list of tokens
    Returns:
        result (list): list of tokens with stopwords removed
    '''
    stop_words = stopwords.words('english')
    stop_words.extend(['chicago', 'hotel', 'would', 'could', 'should', 'might', 'room', 'stay'])
    result = [word for word in words if word not in stop_words]
    return result


# Remove Punctuation and turn all letter to lowercase
def preprocess_ngram(df):
    '''
    Preprocess text reviews into bag-of-words with the following cleaning process: -
        1. Convert to lower case
        2. Remove contractions
        3. Remove punctuations
        4. Lemmatize words
        5. Remove stopwords
    Args:
        df (pd.DataFrame): DataFrame containing text review in column Ori_Review
    Returns
        df_pp (pd.DataFrame): DataFrame containing the same data as input, but add 3 columns
            1. Clean_Review (str): cleaned review texts
            2. PP_Review (str): cleaned review text with stopwords removed
            3. Word_List (list): list of token containing cleaned reviews with stopwords removed
    '''
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


def df2matrix(df, word2ind):
    '''
    Create bag-of-word matrix
    Args:
        df (pd.DataFrame): dataframe containing columns - Ngram, Rating, and Label
        word2ind (dict): dict mapping ngrams to its index
    Returns:
        X (pd.DataFrame): dataframe containing 
            - count of number of each ngrams in each reviews
            - linguistic features
            - Review's rating
        y (pd.Series): ground truth label
            - 1 denotes deceptive review
            - 0 denotes truthful review
    '''
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
