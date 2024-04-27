import nltk
import pandas as pd
from stop_word_list import *


def bigram_filter(bigram):
    tag = nltk.pos_tag(bigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['NN']:
        return False
    if bigram[0] in stop_word_list or bigram[1] in stop_word_list:
        return False
    if 'n' in bigram or 't' in bigram:
        return False
    if 'PRON' in bigram:
        return False
    return True

# Filter for trigrams with only noun-type structures
def trigram_filter(trigram):
    tag = nltk.pos_tag(trigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['JJ','NN']:
        return False
    if trigram[0] in stop_word_list or trigram[-1] in stop_word_list or trigram[1] in stop_word_list:
        return False
    if 'n' in trigram or 't' in trigram:
         return False
    if 'PRON' in trigram:
        return False
    return True

def get_bigramms(data):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_documents([comment.split() for comment in data.skill_description])
    finder.apply_freq_filter(20)
    bigram_scores = finder.score_ngrams(bigram_measures.pmi)
    bigram_pmi = pd.DataFrame(bigram_scores)
    bigram_pmi.columns = ['bigram', 'pmi']
    bigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)

    # Can set pmi threshold to whatever makes sense
    filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram: \
                                                      bigram_filter(bigram['bigram']) \
                                                      and bigram.pmi > 3, axis=1)][:500]
    # Filter for bigrams with only noun-type structures
    bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]

    return bigrams

# trigram_scores
def get_trigramms(data):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.collocations.TrigramCollocationFinder.from_documents([comment.split() for comment in data.skill_description])
    finder.apply_freq_filter(20)
    trigram_scores = finder.score_ngrams(trigram_measures.pmi)

    trigram_pmi = pd.DataFrame(trigram_scores)
    trigram_pmi.columns = ['trigram', 'pmi']
    trigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)

    filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram: \
                                                         trigram_filter(trigram['trigram']) \
                                                         and trigram.pmi > 3, axis=1)][:500]
    trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if
                len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]

    return trigrams