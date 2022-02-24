import nltk

def find_universal_ngrams(documents):
    words = ' '.join(documents.tolist()).split(' ')
    word_list = [document.split(' ') for document in documents]
    
    unigrams = []
    word_freq = {}
    unigrams_dict = {}
    for word in words:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
    for word, freq in word_freq.items():
        if freq >= 16:
            unigrams.append(word)
            unigrams_dict[word] = 0
            
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigramFinder = nltk.collocations.BigramCollocationFinder.from_documents(word_list)
    bigrams = []
    bigrams_dict = {}
    bi_freq_dict = dict(bigramFinder.ngram_fd)
    for bigram in bigramFinder.above_score(bigram_measures.pmi, 0):
        if bi_freq_dict[bigram] >= 16:
            bigrams.append(bigram)
            bigrams_dict[bigram] = 0

    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_documents(word_list)
    trigrams = []
    trigrams_dict = {}
    tri_freq_dict = dict(trigramFinder.ngram_fd)
    for trigram in trigramFinder.above_score(trigram_measures.pmi, 0):
        if tri_freq_dict[trigram] >= 10:
            trigrams.append(trigram)
            trigrams_dict[trigram] = 0
    return unigrams, bigrams, trigrams, unigrams_dict, bigrams_dict, trigrams_dict

    
def find_ngrams(words, unigrams_dict, bigrams_dict, trigrams_dict):
    unigrams = []
    for word in words:
        if word in unigrams_dict:
            unigrams.append(word)

    bigrams = []
    for bigram in zip(words[:-1], words[1:]):
        if bigram in bigrams_dict:
            bigrams.append(' '.join(bigram))

    trigrams = []
    for trigram in zip(words[:-2], words[1:-1], words[2:]):
        if trigram in trigrams_dict:
            trigrams.append(' '.join(trigram))
    return unigrams + bigrams + trigrams