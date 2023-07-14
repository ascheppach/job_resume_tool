import numpy as np
from sklearn.cluster import KMeans
import spacy
import nltk
import re
import pandas as pd
import os


################################################## Step 1: Get Skill data ##########################################
# 1. Get the data

folder_path = 'C:/Users/SEPA/lanchain_ir2/Tech_data/ChatGPT_jira_stories'  # Replace with the path to your folder
file_list = []

# Iterate over each file in the folder and append its content
for filename in os.listdir(folder_path):
    # print(filename)
    if filename.endswith('.txt'):
    #if filename == 'cloud.txt' or filename == 'DataScience.txt':
        # print(filename)
        file_path = os.path.join(folder_path, filename)

        # Open the file and read its contents
        with open(file_path, 'r') as file:
            content = file.read()
            file_list.append(content)

# create own examples for each Jira story (each new Title)
all_files = []
for file in file_list:
    sample_list = file.split('Title')
    all_files += [sample.replace("\nDescription: ", ". ") for sample in sample_list]


# 2. load our custom NER model

nlp_ner = spacy.load("C:/Users/SEPA/lanchain_ir2/model-best")

#test = all_files[1]
#test = 'I have several years of experience with NLP and MLOps. Here I implemented a Text Classification Algorithm with BERT Algorithm. Moreover I have worked with AWS, Kubernetes and Docker.'
extracted_entities_all = []
for sample in all_files:
    doc = nlp_ner(sample)
    extracted_skills = []
    for ent in doc.ents:
        extracted_skills.append(ent.text)
    if extracted_skills:
        # Code to be executed if the list is not empty
        extracted_skills = [element.replace(" ", "_").replace("-", "_") for element in extracted_skills] # build bigrams, trigramms automatically
        extracted_entities_all.append(' '.join(extracted_skills))
    else:
        continue
# extracted_entities_all[1]
# build a data frame with this
# consists of 494 examples
df = pd.DataFrame(extracted_entities_all)
df = df.rename(columns={0: 'skill_description'})
df = df[df['skill_description'] != '']

from stop_word_list import *
from cleanText import *
from cleanText import clean_skills
# In the subsequent step, we will perform data cleaning by converting all characters to lowercase.
# Furthermore, we will remove newline characters and punctuations (e.g., .,/) from the text.
# Additionally, we will apply word lemmatization, which converts words like "running" or "ran" to their base form, such as "run".

clean_df = clean_skills(df, 'skill_description')
print(clean_df.iloc[0][0])


####### bigram and trigram can be skipped and as NER Algorithm has already detected those ###########
# bigram_scores
df = pd.DataFrame(all_files)
df = df.rename(columns={0: 'skill_description'})
df = df[df['skill_description'] != '']

for index, row in df.iterrows():
    # print(df.iloc[index,0])
    row['skill_description'] = row['skill_description'].replace("\n", " ")
    row['skill_description'] = row['skill_description'].replace("Description:", "")

clean_df = clean_skills(df, 'skill_description')

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
                                              and bigram.pmi > 3, axis = 1)][:500]

filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram: \
                                                 trigram_filter(trigram['trigram'])\
                                                 and trigram.pmi > 3, axis = 1)][:500]


bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]
trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]
# examples of bigrams
print(bigrams[:20])

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

# haben ja bigramms schon durch unser named_entity_recognition (welchen wir noch durch unseren overlap finder weiter bereinigen müssen)
# dann haber wir durch unseren bigram algorithmus noch bigrams erstellt.



from gensim import corpora
from collections import Counter
import itertools
series = clean_df['skill_description']
split_series = series.str.split(' ')
skill_list = list(split_series)
merged_list = list(itertools.chain(*skill_list))
skill_frequency = Counter(merged_list)
unique_skills = list(skill_frequency.keys())
transformed_list = [string.replace('_', ' ') for string in unique_skills]

# 1. Abkürzungen identifizieren
# 1.1 bigramms und trigramm erstellen auf basis des gesamten Textes
# 1.2 für jeden dieser bigrams, trigrams

# ['google cloud platform', 'gcp pipeline', 'amazon web services', 'aws ec2', 'python']
#should result in ['google cloud platform', 'google cloud platform pipeline', 'amazon web services', 'amazon web services ec2', 'python']
#original_list = ['google cloud platform', 'gcp pipeline', 'amazon web services', 'aws ec2', 'python']
#original_list = ['google cloud platform', 'gcp pipeline', 'amazon web services', 'aws ec2', 'python']
mapping = {}

transformed_list = []

bigrams
for string in bigrams:
    words = string.split()
    transformed_words = []
    for word in words:
        if any(keyword in word.lower() for keyword in mapping.keys()):
            for keyword in mapping.keys():
                if keyword in word.lower():
                    transformed_words.append(mapping[keyword])
                    break
        else:
            transformed_words.append(word)
            mapping[word.lower()] = word
    transformed_string = ' '.join(transformed_words)
    transformed_list.append(transformed_string)

print(transformed_list)



#original_list = ['google cloud platform', 'gcp', 'amazon web services', 'aws ec2', 'python']
mapping = {}
def create_abbreviating_dictionary(mapping, list):
    for string in list:
        # print(string)
        # string = 'aws ec2' 'aws ec2'
        words = string.split()
        if len(words) > 1:
            abbreviation = ''.join(word[0] for word in words) # nimmt immer ersten buchstaben um abkürzung zu bauen
            mapping[abbreviation.lower()] = string
    return mapping

mapping = create_abbreviating_dictionary(mapping, trigrams)



transformed_list = [mapping.get(string.lower(), string) for string in original_list]
print(transformed_list)


# 2. Synonyme identifizieren
# Data Scientist selbe wie data science -> durch custom trained_embeddings und similarity scores mit threshold
# Mit ChatGPT probieren, input geben und er soll synonyme zusammenfassen
# Am besten: Machine Learning pipeline, Machine Learning models oder Microsoft Azure oder Azure Cloud könnte ich dadurch
# vereinen, indem ich erstmal "gemeinsamen Nenner" finde (Azure oder Machine Learning) un dann restliche
# Wörter rausschmeiße (pipeline, models, Microsoft und Cloud)

def cluster_overlapping_strings(string_list):
    clusters = []
    while string_list:
        current_string = string_list.pop(0)
        current_cluster = [current_string]
        overlapping_strings = set(current_string.split())
        i = 0
        while i < len(string_list):
            if any(word in overlapping_strings for word in string_list[i].split()):
                current_cluster.append(string_list.pop(i))
            else:
                i += 1
        clusters.append(current_cluster)
    return clusters

result_list = cluster_overlapping_strings(transformed_list)
# muss nur noch darauf achten, dass es auch duplicate geben kann wie "Azure cloud" d.h. durch random prinzip einem zuordnen