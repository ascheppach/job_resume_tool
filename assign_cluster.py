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
import ast

import json

folder_path = 'C:/Users/SEPA/lanchain_ir2/Tech_data/ChatGPT_jira_stories'  # Replace with the path to your folder
from helper_functions import open_folder
all_files = open_folder(folder_path)


####################################### Create bigrams and trigrams in order to create abbreviation mapping ######################

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
# Data Scientist selbe wie data science -> durch custom trained_embeddings und similarity scores mit threshold
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


# 2. Dann für meinen CV NER applien

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
# Data Scientist selbe wie data science -> durch custom trained_embeddings und similarity scores mit threshold
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


########################################################### ab hier ####################################################
filename = 'C:/Users/SEPA/lanchain_ir2/my_cluster_dict_new.json'
with open(filename, 'r') as json_file:
    my_cluster_dict = json.load(json_file)

nlp_ner = spacy.load("C:/Users/SEPA/topic_modeling/model-best")
file_path = 'C:/Users/SEPA/lanchain_ir2/CV_Scheppach_text.txt'
with open(file_path, 'r') as file:
    content = file.read()
doc = nlp_ner(content)
extracted_skills_cv = []
for ent in doc.ents:
    extracted_skills_cv.append(ent.text)
print(extracted_skills_cv)

# 3. Danach meinen CV zu den clustern entweder assignen, oder neuen cluster hinzufügen.
import openai
# erstmal aus dem dictionary lediglich die cluster rausextrahieren und eben nicht die einzelnen werte
merged_clusters = list(my_cluster_dict.keys())
sub_clusters = list(list(my_cluster_dict[str(dict)].keys()) for dict in my_cluster_dict)
merged_sub_clusters = [item for sublist in sub_clusters for item in sublist]
merged_skills = []
for category in my_cluster_dict.values():
    for subcategory in category.values():
        merged_skills.extend(subcategory)
print(merged_skills)
all_values = merged_clusters + merged_sub_clusters + merged_skills

# run examples

cluster_question_1 = "Please assign the skill {} to a skillcluster from following list: {}. It can also be assigned to multiple clusters."
example_question_1 = cluster_question_1.format('Machine Learning', json.dumps(merged_sub_clusters))
example_answer_1 = "[{'Machine Learning': []}]"
# Potentiell wäre das hier möglich aber lieber immer nur Überbegriff bei solchen Fällen: Cluster 'Machine Learning' ; Subcluster 'Big Data': {'Machine Learning'} ; Subcluster 'Data Analysis': {'Machine Learning'}
# d.h. als regel soll er mir nur das oberste zurückgeben

# Case es gibt ihn und es ist ein Skill und kommt mehrmals vor in verschiedenen Cluster-Subcluster chains
example_question_2 = cluster_question_1.format('Object Detection', json.dumps(merged_sub_clusters))
example_answer_2 = "[{'Machine Learning': {'Computer Vision': ['Object Detection']}}, {'Artificial Intelligence': {'Computer Vision': ['Object Detection']}}]"

# Case es ist ein Subcluster und wir geben 2er Chain zurück
example_question_3 = cluster_question_1.format('Deep Learning', json.dumps(merged_sub_clusters))
example_answer_3 = "[{'Machine Learning': {'Deep Learning': []}}]"

example_question_4 = cluster_question_1.format('Ruby', json.dumps(merged_sub_clusters))
example_answer_4 = "[{'Software Development': {'Backend Development': ['Ruby']}}]"

cluster_question_2 = "Please assign the skill {} to a skillcluster from following list: {}. It can also be assigned to multiple clusters. If the skill is a synonym of a skillcluster from the list, then return the skillcluster from the list. If the term is not a skill then return 'Not Provided'."
example_question_5 = cluster_question_2.format('Statistics', json.dumps(merged_sub_clusters))
example_answer_5 = "[{'Data Analysis': {'Statistical Analysis': []}}]"

example_question_6 = cluster_question_2.format('Cloud', json.dumps(merged_sub_clusters))
example_answer_6 = "[{'Cloud Services': []}]"

##### Ist eher ein Subcluster als skill (DataScience) -
# in Artificial Intelligence einordnen
example_question_7 = cluster_question_2.format('Data Science', json.dumps(merged_sub_clusters))
example_answer_7 = "[{'Artificial Intelligence': {'Data Science': []}}]"

##### Man kann es einordnen als subcluster (MLOps)
example_question_8 = cluster_question_2.format('MLOps', json.dumps(merged_sub_clusters))
example_answer_8 = "[{'Machine Learning': {'MLOps' : []}}, {'DevOps': {'MLOps' : []}}, {'Software Development': {'MLOps' : []}}]"

##### Man kann es einordnen in mehreren (AutoML)
example_question_9 = cluster_question_2.format('AutoML', json.dumps(merged_sub_clusters))
example_answer_9 = "[{'Machine Learning': {'AutoML': []}}]"

# synonym examples
example_question_synonym_1 = cluster_question_1.format('Azure Services', json.dumps(merged_sub_clusters))
example_answer_synonym_1 = "[{'Cloud Services': {'Azure': []}}]"

example_question_synonym_2 = cluster_question_1.format('NLP algorithms', json.dumps(merged_sub_clusters))
example_answer_synonym_2 = "[{'Artificial Intelligence': {'Natural Language Processing': []}},{'Machine Learning': {'Natural Language Processing': []}}]"

# abbreviation examples
example_question_abbreviation_1 = cluster_question_1.format('Amazon Web Services', json.dumps(merged_sub_clusters))
example_answer_abbreviation_1 = "[{'Cloud Services': {'AWS': []}}]"

example_question_abbreviation_2 = cluster_question_1.format('NLP', json.dumps(merged_sub_clusters))
example_answer_abbreviation_2 = "[{'Artificial Intelligence': {'Natural Language Processing': []},{'Machine Learning': {'Natural Language Processing': []}]"


cluster_chain_cv = []
no_skills = []
for skill in extracted_skills_cv:
    print(skill)
    # skill = extracted_skills_cv[4]
    # skill = 'Machine Learning'
    # skill = 'ticket'
    if skill in all_values:
        print('Skill exists.')
        final_question = cluster_question_1.format(skill, json.dumps(merged_sub_clusters))
        completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for clustering und summarizing terms."},
                {"role": "user", "content": example_question_1},
                {"role": "assistant", "content": example_answer_1},
                {"role": "user", "content": example_question_2},
                {"role": "assistant", "content": example_answer_2},
                {"role": "user", "content": example_question_3},
                {"role": "assistant", "content": example_answer_3},
                {"role": "user", "content": example_question_4},
                {"role": "assistant", "content": example_answer_4},
                {"role": "user", "content": example_question_synonym_1},
                {"role": "assistant", "content": example_answer_synonym_1},
                {"role": "user", "content": example_question_synonym_2},
                {"role": "assistant", "content": example_answer_synonym_2},
                {"role": "user", "content": example_question_abbreviation_1},
                {"role": "assistant", "content": example_answer_abbreviation_1},
                {"role": "user", "content": example_question_abbreviation_2},
                {"role": "assistant", "content": example_answer_abbreviation_2},

                {"role": "user", "content": final_question}
            ]
        )
    else:
        final_question = cluster_question_2.format(skill, json.dumps(merged_sub_clusters))
        completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for clustering und summarizing terms."},
                {"role": "user", "content": example_question_5},
                {"role": "assistant", "content": example_answer_5},
                {"role": "user", "content": example_question_6},
                {"role": "assistant", "content": example_answer_6},
                {"role": "user", "content": example_question_7},
                {"role": "assistant", "content": example_answer_7},
                {"role": "user", "content": example_question_8},
                {"role": "assistant", "content": example_answer_8},
                {"role": "user", "content": example_question_9},
                {"role": "assistant", "content": example_answer_9},

                {"role": "user", "content": example_question_synonym_1},
                {"role": "assistant", "content": example_answer_synonym_1},
                {"role": "user", "content": example_question_synonym_2},
                {"role": "assistant", "content": example_answer_synonym_2},
                {"role": "user", "content": example_question_abbreviation_1},
                {"role": "assistant", "content": example_answer_abbreviation_1},
                {"role": "user", "content": example_question_abbreviation_2},
                {"role": "assistant", "content": example_answer_abbreviation_2},

                {"role": "user", "content": final_question}
            ]
        )

    message = completions.choices[0].message.content
    if message == 'Not Provided':
        no_skills.append(skill)
    else:
        cluster_chain = ast.literal_eval(message)
        cluster_chain_cv.append(cluster_chain)


# storing cluster_dict
filename = 'scheppach_skill_new.json'
with open(filename, 'w') as json_file:
    json.dump(cluster_chain_cv, json_file)

with open(filename, 'r') as json_file:
    my_dict_loaded = json.load(json_file)

data = my_dict_loaded
new_data = []
for sublist in data:
    if len(sublist) > 1:
        for item in sublist:
            new_data.append([item])
    else:
        new_data.append(sublist)

grouped_data = {}
for sublist in new_data:
    # sublist = data[0]
    highest_level = None
    for item in sublist:
        # item = sublist[0]
        highest_level = next(iter(item))
        break  # We only need the highest level key
    if highest_level:
        if highest_level not in grouped_data:
            grouped_data[highest_level] = []
        grouped_data[highest_level].extend(sublist)

# new_grouped_data = []
final_dict = {}
for group in grouped_data.items():
    # group_check.append(group[1])
    # print(list(group[1]))
    group = group[1]
    unique_elements = set()
    filtered_list = []
    for item in group: # iteriert über: [{'Data Analysis': {'Statistical Analysis': []}}, {'Data Analysis': {'Statistical Analysis': []}}, {'Data Analysis': {'Data Scientist': []}}]
        # Convert the dictionary to a string to check for uniqueness
        item_str = str(item)
        if item_str not in unique_elements:
            unique_elements.add(item_str)
            filtered_list.append(item)
    # new_grouped_data.append(filtered_list)
    transformed_data = {}
    # key_t = list(data[0].keys())
    for item in filtered_list:
        # print(item)
        # item['Machine Learning']
        for key, value in item.items():
            # print(key) # mit dem key ein neues dict erstellen
            if value == []:
                continue
            # print(value)
            for key_sub, sub_clust in value.items():
                print(key_sub)
                print(sub_clust)
                if key_sub not in transformed_data:
                    transformed_data[key_sub] = sub_clust
                else:
                    transformed_data[key_sub].append(sub_clust[0])
    final_dict[key] = transformed_data


