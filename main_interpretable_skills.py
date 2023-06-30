import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import spacy
import nltk
import re
import string
import pandas as pd
import numpy as np
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#stop_word_list = stopwords.words('english')

from stop_word_list import *
from cleanText import *
import gensim
from gensim import corpora
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import json
# %matplotlib inline

# Load the data
import os

folder_path = 'C:/Users/SEPA/topic_modeling/Tech_data/ChatGPT_jira_stories'  # Replace with the path to your folder
file_list = []

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # print(filename)
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        # Open the file and read its contents
        with open(file_path, 'r') as file:
            content = file.read()
            file_list.append(content)

all_files = []
for file in file_list:
    all_files += file.split('Title')

df = pd.DataFrame(all_files)
df = df.rename(columns={0: 'skill_description'})
df = df[df['skill_description'] != '']

for index, row in df.iterrows():
    # print(df.iloc[index,0])
    row['skill_description'] = row['skill_description'].replace("\n", " ")
    row['skill_description'] = row['skill_description'].replace("Description:", "")

clean_df = clean_all(df, 'skill_description')
clean_df.head()


# punctuation noch entfernen
# /n und /n/n noch entfernen
# Description noch entfernen


# bigram_scores
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_documents([comment.split() for comment in clean_df.skill_description])
finder.apply_freq_filter(20)
bigram_scores = finder.score_ngrams(bigram_measures.pmi)

bigram_pmi = pd.DataFrame(bigram_scores)
bigram_pmi.columns = ['bigram', 'pmi']
bigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)

# trigram_scores
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = nltk.collocations.TrigramCollocationFinder.from_documents([comment.split() for comment in clean_df.skill_description])
finder.apply_freq_filter(20)
trigram_scores = finder.score_ngrams(trigram_measures.pmi)

trigram_pmi = pd.DataFrame(trigram_scores)
trigram_pmi.columns = ['trigram', 'pmi']
trigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)

# Filter for bigrams with only noun-type structures
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
# Can set pmi threshold to whatever makes sense - eyeball through and select threshold where n-grams stop making sense
# choose top 500 ngrams in this case ranked by PMI that have noun like structures
filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram: \
                                              bigram_filter(bigram['bigram'])\
                                              and bigram.pmi > 5, axis = 1)][:500]

filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram: \
                                                 trigram_filter(trigram['trigram'])\
                                                 and trigram.pmi > 5, axis = 1)][:500]


bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]
trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]
# examples of bigrams
bigrams[:10]

# Concatenate n-grams
def replace_ngram(x):
    for gram in trigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    return x
reviews_w_ngrams = clean_df.copy()
reviews_w_ngrams.skill_description = reviews_w_ngrams.skill_description.map(lambda x: replace_ngram(x))
# tokenize reviews + remove stop words + remove names + remove words with less than 2 characters
reviews_w_ngrams = reviews_w_ngrams.skill_description.map(lambda x: [word for word in x.split()\
                                                 if word not in stop_word_list\
                                                              and word not in english_names\
                                                              and len(word) > 2])
reviews_w_ngrams.head()


############################################# Filter for only nouns ####################################################
def noun_only(x):
    pos_comment = nltk.pos_tag(x)
    filtered = [word[0] for word in pos_comment if word[1] in ['NN']]
    # to filter both noun and verbs
    #filtered = [word[0] for word in pos_comment if word[1] in ['NN','VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
    return filtered
final_reviews = reviews_w_ngrams.map(noun_only)

# final_reviews.iloc[0]
# df[0]

########################################## LDA Model ###################################################################
# get unique words or tokens present in the text corpus
dictionary = corpora.Dictionary(final_reviews)
# The doc2bow converts a document into bag-of-words
# takes a document as input and returns a sparse vector,
# each element of  the vector represents a unique word in the dictionary and its corresponding frequency in the document.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in final_reviews]



coherence = []
for k in range(5, 25): # iteriere über 5-25 anzahl topics und berechne für jeden die coherence
    print('Round: ' + str(k))
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=k, id2word=dictionary, passes=40, \
                   iterations=200, chunksize=10000, eval_every=None)

    cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=final_reviews, \
                                                     dictionary=dictionary, coherence='c_v')
    coherence.append((k, cm.get_coherence()))


# evaluiere coherence
x_val = [x[0] for x in coherence]
y_val = [x[1] for x in coherence]
plt.plot(x_val,y_val)
plt.scatter(x_val,y_val)
plt.title('Number of Topics vs. Coherence')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence')
plt.xticks(x_val)
plt.show()


# er geht 40 mal über gesamten/alle chunks. uns für jedes dokument macht er 200iterations
# Passes: The number of times model iterates through the whole corpus
# Iterations: The number of iterations the model trains on each pass
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=6, id2word = dictionary, passes=40,\
               iterations=200,  chunksize = 100, eval_every = None, random_state=0)

Lda2 = gensim.models.ldamodel.LdaModel
ldamodel2 = Lda2(doc_term_matrix, num_topics=20, id2word = dictionary, passes=40,\
               iterations=200,  chunksize = 100, eval_every = None, random_state=0)


ldamodel.show_topics(6, num_words=10, formatted=False)
# ldamodel2.show_topics(20, num_words=10, formatted=False)


# visualization with pyLDAvis
topic_data = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, mds = 'pcoa', num_terms=6)
# pyLDAvis.display(topic_data)
pyLDAvis.save_html(topic_data, 'topic_visualization.html')


all_topics = {}
num_terms = 6 # Adjust number of words to represent each topic
lambd = 0.2 # Adjust this accordingly based on tuning above
# there is the risk, that words are globally important. a low lambda value can correct that issue, by giving more
# importance to topic exclusivity
topics = []
top_words = []
relevance = []
for i in range(1,7): #Adjust this to reflect number of topics chosen for final LDA model
    topic = topic_data.topic_info[topic_data.topic_info.Category == 'Topic'+str(i)].copy()
    topic['relevance'] = topic['loglift']*(1-lambd)+topic['logprob']*lambd

    top_words.append(topic.sort_values('relevance', ascending=False).loc[:, ['Term']])
    relevance.append(topic.sort_values('relevance', ascending=False).loc[:, ['relevance']])
    topics.append('Topic' + str(i))
    all_topics['Topic '+str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:num_terms].values
checkData = pd.DataFrame(all_topics).T


import matplotlib.pyplot as plt

# Example data
import matplotlib.pyplot as plt

# Create subplots
fig, axs = plt.subplots(1, 6, figsize=(15, 5))

# Iterate over each topic and create a bar plot in the corresponding subplot
for i, topic in enumerate(topics):
    axs[i].barh(list(top_words[i].iloc[:, 0])[:num_terms], list(relevance[i].iloc[:, 0])[:num_terms])
    axs[i].set_ylabel('Words')
    axs[i].set_xlabel('Relevance')
    axs[i].set_title(f'Top Words - {topic}')
    axs[i].invert_yaxis()

# Adjust spacing between subplots
plt.tight_layout()

# Display the combined plot
plt.show()




