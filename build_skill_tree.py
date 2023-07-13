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


from gensim import corpora
from collections import Counter
import itertools


series = df['skill_description']
split_series = series.str.split(' ')
skill_list = list(split_series)
merged_list = list(itertools.chain(*skill_list))
skill_frequency = Counter(merged_list)
unique_skills = list(skill_frequency.keys())

# 1. Synonyme identifizieren
# Data Scientist selbe wie data science -> durch custom trained_embeddings und similarity scores mit threshold
# Mit ChatGPT probieren, input geben und er soll synonyme zusammenfassen
# Am besten: Machine Learning pipeline, Machine Learning models oder Microsoft Azure oder Azure Cloud könnte ich dadurch
# vereinen, indem ich erstmal "gemeinsamen Nenner" finde (Azure oder Machine Learning) un dann restliche
# Wörter rausschmeiße (pipeline, models, Microsoft und Cloud)
test_list = ['Microsoft Azure','Azure Cloud']
def find_overlapping_strings(string_list):
    overlapping_strings = []
    for i in range(len(string_list)):
        for j in range(i + 1, len(string_list)):
            common_substring = set(string_list[i].split()) & set(string_list[j].split())
            overlapping_strings.extend(common_substring)
    return list(set(overlapping_strings))

string_list = ['Machine Learning pipeline','Machine Learning model','Microsoft Azure', 'Azure Cloud', 'AWS', 'Cloud', 'Python']
overlapping_strings = find_overlapping_strings(string_list)
print(overlapping_strings)
# funktioniert schon sehr gut:
# muss nur noch darauf achten, dass es auch duplicate geben kann wie "Azure cloud" d.h. durch random prinzip einem zuordnen
# Machine Learning wäre durch bigram erstmal ja auch als ein wort erkannt worden

# jetzt muss ich noch dafür sorgen, dass wenn
# I have following list string_list = ['Machine Learning pipeline','Machine Learning model']. Now I want to replace each element
# if they it also exists in this list ['Machine', 'Azure']. So the result would be to return me following list ['Machine', 'Machine'].
import re

string_list = ['Machine Learning pipeline', 'Machine Learning model']
replace_list = ['Machine', 'Azure']

overlap_list = []

for string in string_list:
    overlap_strings = [word for word in replace_list if re.search(r'\b{}\b'.format(word), string)]
    overlap_list.extend(overlap_strings)

print(overlap_list)

# 2. Abkürzungen identifizieren
# 2.1. Abbkürzungen finden: Alle Wörter, bei denen alle Buchstaben groß geschrieben sind: ML oder AWS
# 2.2 Lange Schreibweise dazu finden: Den bigramm und trigramm Teil aus Machine Learning
# Am besten: zuerst bigram und trigram herausfinden und dann einfach nur die Anfangsbuchstaben nehmen und als Abkürzung
#            definieren

# 3. Manuell eine Synonym Datenbank erstellen








dictionary = corpora.Dictionary(split_series)
for pot_skil in dictionary.token2id:
    print(pot_skil)