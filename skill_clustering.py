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
    #if filename.endswith('.txt'):
    if filename == 'cloud.txt' or filename == 'DataScience.txt':
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


# vermutlich sollte ich doch oben nicht direkt die bigrams bauen, sondern erst hier (weil es bigrams bildet, nur wenn sie
# in der form auch gehäuft vorkommen)
from gensim import corpora

series = df['skill_description']
split_series = series.str.split(' ')
dictionary = corpora.Dictionary(split_series)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in split_series]
import gensim
import matplotlib.pyplot as plt

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=100,\
               iterations=300, chunksize = 50, eval_every = None, random_state=0)
import pyLDAvis.gensim
# visualization with pyLDAvis
topic_data = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, mds = 'pcoa')#,num_terms=6)
# pyLDAvis.display(topic_data)
pyLDAvis.save_html(topic_data, 'topic_visualization.html')

num_words = 10  # Number of most frequent words to retrieve per topic
topics = ldamodel.show_topics(num_topics=2, num_words=num_words, formatted=False)
# topic0 is NLP (plot 2); topic1 DL MLOps (plot 3) ; topic2 DL ML (plot 1)

# Empty data frame with column names for topics
topic_columns = [f'Topic {i}' for i in range(ldamodel.num_topics)]
df = pd.DataFrame(columns=topic_columns)

# Iterate over documents and add topic distribution as rows
i=0
for doc in doc_term_matrix:
    topic_dist = ldamodel.get_document_topics(doc, minimum_probability=0)
    topic_probabilities = [prob for _, prob in topic_dist]
    df.loc[len(df)] = topic_probabilities
topics_all = df

# Data for the pie chart
labels = ['DataScientist', 'Cloud']
print(extracted_entities_all[0])# example document
sizes = topics_all.iloc[0] # Represents the percentage of each slice
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Skill Distribution')
plt.show()


########################################################### Clustering ####################################################
# Now, we will utilize the feature matrix derived from our LDA topic modeling algorithm to construct our clustering. There we
# increase the number of topic to 5 to have more granular features available for our clustering.
# This feature matrix effectively captures the co-occurrence of words and the document's contribution to each topic.
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=100,\
               iterations=300, chunksize = 50, eval_every = None, random_state=0)
topic_columns = [f'Topic {i}' for i in range(ldamodel.num_topics)]
df = pd.DataFrame(columns=topic_columns)
i=0
for doc in doc_term_matrix:
    topic_dist = ldamodel.get_document_topics(doc, minimum_probability=0)
    topic_probabilities = [prob for _, prob in topic_dist]
    df.loc[len(df)] = topic_probabilities
topics_all = df


from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
plt.figure()
plt.title("Dendrograms")
Z = hierarchy.linkage(topics_all, method='single')
dend = hierarchy.dendrogram(Z)
plt.show()

from scipy.cluster.hierarchy import cut_tree
clusters = cut_tree(Z, n_clusters=range(1, 5)) # topics_all.shape[0]))

# first split (with 2 clusters)
cluster_0, cluster_1 = [], []
for i, cluster in enumerate(clusters[:,1]):
    #print(i)
    if cluster == 0:
        cluster_0.append(extracted_entities_all[i])
    else:
        cluster_1.append(extracted_entities_all[i])