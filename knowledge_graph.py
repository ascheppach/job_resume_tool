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


#################################### build knowledge graph ################################
job_net = Network(height='1000px', width='100%', bgcolor='#222222', font_color='white')

job_net.barnes_hut()
sources = data_graph['Job ID']
targets = data_graph['skills']
values=data_graph['years skills']
sources_resume = data_graph_resume['document']
targets_resume = data_graph_resume['skills']

edge_data = zip(sources, targets, values )
resume_edge=zip(sources_resume, targets_resume)
for j,e in enumerate(edge_data):
    src = e[0]
    dst = e[1]
    w = e[2]


    job_net.add_node(src, src, color='#dd4b39', title=src)
    job_net.add_node(dst, dst, title=dst)


    if str(w).isdigit():
        if w is None:

            job_net.add_edge(src, dst, value=w, color='#00ff1e', label=w)
        if 1<w<=5:
            job_net.add_edge(src, dst, value=w, color='#FFFF00', label=w)
        if w>5:
            job_net.add_edge(src, dst, value=w, color='#dd4b39', label=w)

    else:
        job_net.add_edge(src, dst, value=0.1, dashes=True)
for j,e in enumerate(resume_edge):
    src = 'resume'
    dst = e[1]

    job_net.add_node(src, src, color='#dd4b39', title=src)
    job_net.add_node(dst, dst, title=dst)
    job_net.add_edge(src, dst, color='#00ff1e')
neighbor_map = job_net.get_adj_list()
for node in job_net.nodes:
    node['title'] += ' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
    node['value'] = len(neighbor_map[node['id']])
# add neighbor data to node hover data
job_net.show_buttons(filter_=['physics'])
job_net.show('job_knolwedge_graph.html')

