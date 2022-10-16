import pandas as pd


def ling_feature(df):
    '''
    Extract the following linguistic features:
    Args:
        df (pd.DataFrame): dataframe containing columns 'Ori_Review', 'Clean_Review' and 'Word_list'
    Returns:
        df_new (pd.DataFrame): dataframe containing the data ss input df and additional linguistic feature columns
            - num_word: number of words in the review
            - num_coreword: number of core words in the review
            - num_stopword: number of stop words in the review
            - num_char: number of characters in the review
            - char_per_word: average word length of the review
            - num_first_sing: number of first-person singular pronoun in the review
            - num_first_pru: number of first-person plural pronoun in the review
            - num_thirs: number of third-person pronoun in the review
    '''
    df_new = df.copy()
    words = df_new['Clean_Review'].apply(lambda x: x.split())
    df_new['num_word'] = words.apply(len)
    df_new['num_coreword'] = df_new['Word_List'].apply(len)
    df_new['num_stopword'] = df_new['num_word'] - df_new['num_coreword']
    df_new['num_char'] = df_new['Ori_Review'].apply(len)
    df_new['char_per_word'] = df_new['num_char'] / df_new['num_word']
    first_sing = ['i', 'me', 'my', 'mine', 'myself']
    first_pru = ['we', 'our', 'us', 'ours', 'ourselves']
    third = ['he', 'she', 'they', 'his', 'her', 'their', 'hers', 'theirs', 'himself', 'herself', 'themselves']
    df_new['num_first_sing'] = words.apply(lambda x: count_pronoun(x, first_sing))
    df_new['num_first_pru'] = words.apply(lambda x: count_pronoun(x, first_pru))
    df_new['num_third'] = words.apply(lambda x: count_pronoun(x, third))
    return df_new

def count_pronoun(words, pronouns):
    '''
    Count number of pronouns from the input words according to the pronouns list
    Args:
        words (list): list of input word 
        pronouns (lsit): list of pronouns we want to count
    Returns:
        result (int): number of pronouns matching those in the pronouns list
    '''
    result = len([x for x in words if x in pronouns])
    return result