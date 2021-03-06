import pandas as pd


def count_pronoun(words, pronouns):
    result = len([x for x in words if x in pronouns])
    return result


def ling_feature(df):
    df_new = df.copy()
    df_new['num_word'] = df_new['Word_List_all'].apply(len)
    df_new['num_coreword'] = df_new['Word_List'].apply(len)
    df_new['num_stopword'] = df_new['num_word'] - df_new['num_coreword']
    df_new['num_char'] = df_new['Ori_Review'].apply(len)
    df_new['char_per_word'] = df_new['num_char'] / df_new['num_word']
    first_sing = ['i', 'me', 'my', 'mine', 'myself']
    df_new['num_first_sing'] = df_new['Word_List_all'].apply(lambda x: count_pronoun(x, first_sing))
    return df_new
