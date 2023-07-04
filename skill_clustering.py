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


# vermutlich sollte ich doch oben nicht direkt die bigrams bauen, sondern erst hier (weil es bigrams bildet, nur wenn sie
# in der form auch gehäuft vorkommen)
from gensim import corpora

series = df['skill_description']
split_series = series.str.split(' ')
dictionary = corpora.Dictionary(split_series)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in split_series]
import gensim
import matplotlib.pyplot as plt

coherence = []
for k in range(5, 25):
    print('Round: ' + str(k))
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=k, id2word=dictionary, passes=40, \
                   iterations=200, chunksize=100, eval_every=None)

    cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=split_series, \
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

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=7, id2word = dictionary, passes=40,\
               iterations=200, chunksize = 1000, eval_every = None, random_state=0)
# ldamodel.show_topics(6, num_words=10, formatted=False)
import pyLDAvis.gensim

# visualization with pyLDAvis
topic_data = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, mds = 'pcoa')#,num_terms=6)
# pyLDAvis.display(topic_data)
pyLDAvis.save_html(topic_data, 'topic_visualization.html')


# Empty data frame with column names for topics
topic_columns = [f'Topic {i}' for i in range(ldamodel.num_topics)]
df = pd.DataFrame(columns=topic_columns)

# Iterate over documents and add topic distribution as rows
i=0
for doc in doc_term_matrix:
    print(doc)
    #i+=1
    #if i >1:
    #    break
    topic_dist = ldamodel.get_document_topics(doc, minimum_probability=0)
    topic_probabilities = [prob for _, prob in topic_dist]
    df.loc[len(df)] = topic_probabilities


split_series[0:5]





topics_all = pd.DataFrame.from_dict(document_topic, orient='index')











nlp = spacy.load('en_core_web_sm')
# Skills data
skills = ["AWS", "Kubernetes", "Azure", "Machine Learning", "Python", "Deep Learning", "AWS S3", "R"]

# Generate word embeddings for each skill
skill_embeddings = [nlp(skill).vector for skill in skills] # er hat 8 vectoren

# Convert the embeddings to a numpy array
X = np.array(skill_embeddings)

# K-means clustering
num_clusters = 2  # Set the number of clusters

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get the cluster assignments for each skill
cluster_labels = kmeans.labels_

# Print the skill clusters
for i in range(num_clusters):
    cluster_skills = [skills[j] for j in range(len(skills)) if cluster_labels[j] == i]
    print(f"Cluster {i + 1}: {cluster_skills}")
