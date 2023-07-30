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


################################################## Step 1: Get Skill data ##########################################
# 1. Get the data

folder_path = 'C:/Users/SEPA/lanchain_ir2/Tech_data/ChatGPT_jira_stories'  # Replace with the path to your folder
all_files = []
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
            content = content.split('Title')
            all_files += [cont.replace("\nDescription: ", ". ") for cont in content]


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
def create_abbreviating_dictionary(mapping, list):
    for string in list:
        words = string.split()
        if len(words) > 1:
            abbreviation = ''.join(word[0] for word in words)
            mapping[abbreviation.lower()] = string
    return mapping

mapping = create_abbreviating_dictionary(mapping, trigrams)
mapping['aws'] = 'amazon web services'
mapping['gke'] = 'google kubernetes engine'
mapping['gcp'] = 'google cloud platform'


#### transform abbreviations to correct name
# 2. load our custom NER model
nlp_ner = spacy.load("C:/Users/SEPA/lanchain_ir2/model-best")
regex = re.compile('[' + re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^`{|}~') + '\\r\\t\\n]')

#test = all_files[1]
#test = 'I have several years of experience with NLP and MLOps. Here I implemented a Text Classification Algorithm with BERT Algorithm. Moreover I have worked with AWS, Kubernetes and Docker.'
extracted_entities_all = []
for sample in all_files: # list und jedes element ist
    # sample = all_files[3]
    doc = nlp_ner(sample)
    extracted_skills = []
    for ent in doc.ents:
        extracted_skills.append(ent.text)
    if extracted_skills: # Code to be executed if the list is not empty
        #extracted_skills = [element.replace(' ', '_').replace("-", "_") for element in extracted_skills] # einzelne listenelemente mit "_"
        #extracted_entities_all.append(' '.join(extracted_skills)) # wird gejoined
        # extracted_skills = extracted_skills.lower()
        # remove punctuation: entferne kommas, slash zeichen usw.
        # df[col_name] = df[col_name].map(punc_skill)
        extracted_skills = [element.lower() for element in extracted_skills]
        extracted_skills = [regex.sub(" ", element) for element in extracted_skills]
        extracted_entities_all.append(extracted_skills)
    else:
        continue

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
def transformAbbreviations(skill_list):
    updated_skill_list = []
    for doc in skill_list:
        # string = 'aws ec2'
        new_term = ''
        for i,str in enumerate(doc.split()):
            if i==0:
                new_term += mapping.get(str.lower(), str)
            else:
                new_term += ' ' + mapping.get(str.lower(), str)
        updated_skill_list.append(new_term)
    return updated_skill_list

from collections import Counter
transformed_skill_list = transformAbbreviations(merged_list)
skill_frequency = Counter(transformed_skill_list)
unique_skills = list(skill_frequency.keys())

# 2. Summarize/build clusters based on overlapping terms
# Data Scientist selbe wie data science -> durch custom trained_embeddings und similarity scores mit threshold
# Mit ChatGPT probieren, input geben und er soll synonyme zusammenfassen
# Am besten: Machine Learning pipeline, Machine Learning models oder Microsoft Azure oder Azure Cloud könnte ich dadurch
# vereinen, indem ich erstmal "gemeinsamen Nenner" finde (Azure oder Machine Learning) und dann restliche
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

result_list = cluster_overlapping_strings(unique_skills)
# muss nur noch darauf achten, dass es auch duplicate geben kann wie "Azure cloud" d.h. durch random prinzip einem zuordnen


# if a listelement/cluster has more than 1 element, ask gpt return you a summarize term
import openai

openai.api_key = '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'
response_list = []
for i, term_cluster in enumerate(result_list):
    flag = True
    while (flag):
        try:
            print(i)
            # if i > 60:
            #    continue
            # term_cluster = string(result_list[2][0])
            if len(term_cluster) > 1:
                summarize_term = "What could be a summarizing term for following word cluster: {}? Please only return the summarizing term and no additional text. If not found, return 'Not a skill'."
                result_string = summarize_term.format(term_cluster)
                completions = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant for clustering und summarizing terms."},
                        {"role": "user", "content": result_string}
                    ]
                )
                message = completions.choices[0].message.content
                if message != 'Not a skill':
                    response_list.append(message)

            else:
                summarize_term = "Is this term describing a technical skill: {}? If not, return 'Not a skill'"
                result_string = summarize_term.format(term_cluster)
                completions = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant for clustering und summarizing terms."},
                        {"role": "user", "content": result_string}
                    ]
                )
                message = completions.choices[0].message.content
                if message != 'Not a skill':
                    response_list.append(term_cluster[0])
            flag = False
        except openai.error.OpenAIError as e:
            print("OpenAI Server Error happened here.")


import csv
with open('response_list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(response_list)

# Opening the CSV file and reading the list back
with open('response_list.csv', 'r') as file:
    reader = csv.reader(file)
    loaded_list = next(reader)

example_response_list = ['EC2', 'S3', 'RDS', 'Lambda','Virtual Machines', 'Blob Storage', 'SQL Database', 'Functions',
                         'Compute Engine', 'Cloud Storage', 'Cloud SQL', 'Cloud Functions',
                         'Jenkins', 'GitLab CI/CD', 'Travis CI', 'Kubernetes', 'Docker Swarm', 'Amazon ECS',
                         'Terraform', 'AWS CloudFormation', 'Azure Resource Manager','NumPy', 'Pandas', 'Scikit-learn',
                         'React', 'Angular', 'Vue.js','SQL', 'GraphQL']

example_clusters = {
    'Cloud Services': {
        'AWS Services': ['EC2', 'S3', 'RDS', 'Lambda'],
        'Azure Services': ['Virtual Machines', 'Blob Storage', 'SQL Database', 'Functions'],
        'Google Cloud Services': ['Compute Engine', 'Cloud Storage', 'Cloud SQL', 'Cloud Functions']
    },
    'DevOps Tools': {
        'CI/CD': ['Jenkins', 'GitLab CI/CD', 'Travis CI'],
        'Container Orchestration': ['Kubernetes', 'Docker Swarm', 'Amazon ECS'],
        'Infrastructure as Code': ['Terraform', 'AWS CloudFormation', 'Azure Resource Manager']
    },
    'Programming Languages': {
        'Python Ecosystem': ['NumPy', 'Pandas', 'Scikit-learn'],
        'JavaScript Frameworks': ['React', 'Angular', 'Vue.js'],
        'Data Query Languages': ['SQL', 'GraphQL']
    }
}

import json
example_clusters_string = json.dumps(example_clusters)

# create skill_cluster based on the summarized terms.
summarize_term = "Please cluster following skills and also include different hierarchy levels or subclusters: {}."
result_string = summarize_term.format(response_list)
# create example clustering
example_result_string = summarize_term.format(example_response_list)
completions = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.0,
    messages=[
        {"role": "system", "content": "You are a helpful assistant for clustering und summarizing terms."},
        {"role": "user", "content": example_result_string},
        {"role": "assistant", "content": example_clusters_string},
        {"role": "user", "content": result_string}
    ]
)
message = completions.choices[0].message.content
my_cluster_dict = json.loads(message)

filename = 'my_cluster_dict.json'
with open(filename, 'w') as json_file:
    json.dump(my_cluster_dict, json_file)

# Read the JSON data from the file
with open(filename, 'r') as json_file:
    my_dict_loaded = json.load(json_file)
