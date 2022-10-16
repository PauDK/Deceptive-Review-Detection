import nltk

def find_universal_ngrams(documents, uni_thres, bi_thres, tri_thres):
    '''
    Create list of vocabs (unigram, bigram, and trigram) from the entire documents. 
    Only keep grams that appears more than a certain number of times.
    Args:
        documents (pd.Series): Series of text reviews
        uni_thres (int): Appearance threshold for a unigram to be included in unigrams
        bi_thres (int): Appearance threshold for a bigram to be included in bigrams
        tri_thres (int): Appearance threshold for a trigram to be included in trigrams
    Returns:
        unigrams (list): list of string unigrams
        bigrams (list): list of 2-element string tuples (bigram) containing word pair that appears together frequently
        trigrams (list): list of 3-element string tuples (trigram) containing a group of 3 words that appear together frequently
    '''
    words = ' '.join(documents.tolist()).split(' ')
    word_list = [document.split(' ') for document in documents]
    
    unigrams = []
    word_freq = {}
    for word in words:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
    for word, freq in word_freq.items():
        if freq >= uni_thres:
            unigrams.append(word)
            
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigramFinder = nltk.collocations.BigramCollocationFinder.from_documents(word_list)
    bigrams = []
    bi_freq_dict = dict(bigramFinder.ngram_fd)
    for bigram in bigramFinder.above_score(bigram_measures.pmi, 0):
        if bi_freq_dict[bigram] >= bi_thres:
            bigrams.append(bigram)

    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_documents(word_list)
    trigrams = []
    tri_freq_dict = dict(trigramFinder.ngram_fd)
    for trigram in trigramFinder.above_score(trigram_measures.pmi, 0):
        if tri_freq_dict[trigram] >= tri_thres:
            trigrams.append(trigram)
    return unigrams, bigrams, trigrams

    
def find_ngrams(words, unigrams, bigrams, trigrams):
    '''
    Extract Ngrams from the input strings based on the unigrams, bigrams, and trigrams vocabulary
    Args:
        words (list): input list of string
        unigrams (list): unigram vocab
        bigrams (list): bigram vocab
        trigrams (list): trigram vocab
    Returns:
        ngrams (list): a list of strings containing unigram, bigram and trigram that's in the input string.
    '''
    uni = []
    for word in words:
        if word in unigrams:
            uni.append(word)

    bi = []
    for bigram in zip(words[:-1], words[1:]):
        if bigram in bigrams:
            bi.append(' '.join(bigram))

    tri = []
    for trigram in zip(words[:-2], words[1:-1], words[2:]):
        if trigram in trigrams:
            tri.append(' '.join(trigram))
    ngrams = uni + bi + tri
    return ngrams