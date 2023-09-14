import numpy as np
from sklearn.cluster import KMeans
import spacy
import nltk
import re
import pandas as pd
import os
from stop_word_list import *
from cleanText import *
from cleanText import clean_skills
import json


################################################## Step 1: Get Skill data ##########################################
# 1. Get the data

folder_path = 'C:/Users/SEPA/lanchain_ir2/Tech_data/ChatGPT_jira_stories'  # Replace with the path to your folder

from helper_functions import open_folder

all_files = open_folder(folder_path)


####################################### Create bigrams and trigrams in order to create abbreviation mapping ######################
### 1. buil bigrams and trigrams

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
from helper_functions import bigram_filter, trigram_filter

# Can set pmi threshold to whatever makes sense
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

### 2. Create abbreviation dictionary
# 1.1 bigramms und trigramm erstellen auf basis des gesamten Textes
# 1.2 für jeden dieser bigrams, trigrams abkürzung erstellen und in mapping hinterlegen
# 1.3 auf basis von originalem text die abkürzungen zu normalem begriff umwandeln
mapping = {}

from helper_functions import create_abbreviating_dictionary

mapping = create_abbreviating_dictionary(mapping, trigrams)
mapping['aws'] = 'amazon web services'
mapping['gke'] = 'google kubernetes engine'
mapping['gcp'] = 'google cloud platform'


#### transform abbreviations to correct name
# 2. load our custom NER model

#test = all_files[1]
#test = 'I have several years of experience with NLP and MLOps. Here I implemented a Text Classification Algorithm with BERT Algorithm. Moreover I have worked with AWS, Kubernetes and Docker.'

from helper_functions import extract_skill_entities

extracted_entities_all = extract_skill_entities(all_files)

# comment = 'Time-Series Analysis Model'

# er wandelt zuerst in aws_ec2 um damit er später split machen kann ; dann wandelt er wieder in aws ec2 um damit er overlapping words machen kann
# df = pd.DataFrame(extracted_entities_all)
# df = df.rename(columns={0: 'skill_description'})
# df = df[df['skill_description'] != '']
# clean_df = clean_skills(df, 'skill_description')
#result = clean_df['skill_description'].apply(lambda x: x.split())

#all_transf_docs = []
#for doc in result:
#    transformed_doc = [string.replace('_', ' ') for string in doc]
#    all_transf_docs.append(transformed_doc) # liste bei der alle listenelemente selber wieder eine liste ist mit ['aws ec2', 'deplyoment', ...]

merged_list = [item for sublist in extracted_entities_all for item in sublist] # damit wir nicht mehr jede dokument eine listenelement ist sonder jeder skill
# ein listenelement

# transform aws to Amazon Web Services

from helper_functions import transformAbbreviations
from collections import Counter
transformed_skill_list = transformAbbreviations(merged_list, mapping)
skill_frequency = Counter(transformed_skill_list)
unique_skills = list(skill_frequency.keys())

# 2. Summarize/build clusters based on overlapping terms
# Data Scientist same as data science -> durch custom trained_embeddings und similarity scores mit threshold
# Mit ChatGPT probieren, input geben und er soll synonyme zusammenfassen
# Am besten: Machine Learning pipeline, Machine Learning models oder Microsoft Azure oder Azure Cloud könnte ich dadurch
# vereinen, indem ich erstmal "gemeinsamen Nenner" finde (Azure oder Machine Learning) und dann restliche
# Wörter rausschmeiße (pipeline, models, Microsoft und Cloud)

from helper_functions import cluster_overlapping_strings

result_list = cluster_overlapping_strings(unique_skills)
# muss nur noch darauf achten, dass es auch duplicate geben kann wie "Azure cloud" d.h. durch random prinzip einem zuordnen

# if a listelement/cluster has more than 1 element, ask gpt return you a summarize term

from helper_functions import summarize_skill_terms
response_list = summarize_skill_terms(result_list)

from helper_functions import create_skill_clusters
my_cluster_dict = create_skill_clusters(response_list)


import csv
# storing response_list
with open('response_list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(response_list)

# Opening the CSV file and reading the list back
with open('response_list.csv', 'r') as file:
    reader = csv.reader(file)
    loaded_list = next(reader)

# storing cluster_dict
filename = 'my_cluster_dict_new.json'
with open(filename, 'w') as json_file:
    json.dump(my_cluster_dict, json_file)

# opening cluster_dict
with open(filename, 'r') as json_file:
    my_dict_loaded = json.load(json_file)
