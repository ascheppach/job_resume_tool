import numpy as np
from sklearn.cluster import KMeans
import spacy
import nltk
import re
import pandas as pd
import os

nlp_ner = spacy.load("C:/Users/SEPA/lanchain_ir2/model-best")

################################################## Step 1: Get Skill data ##########################################
# 1. Define Job-skills

job_skills = ['MLOps', 'AWS', 'NLP', 'ComputerVision', 'DeepReinforcementLearning']


# 2. Get Resume-skills
file_path = 'C:/Users/SEPA/lanchain_ir2/CV_Scheppach_text.txt'

with open(file_path, 'r') as file:
    content = file.read()

doc = nlp_ner(content)

extracted_skills_cv = []
for ent in doc.ents:
    extracted_skills_cv.append(ent.text)
print(extracted_skills_cv)

unique_extracted_skills_cv = list(set(extracted_skills_cv))
skills_cv = pd.DataFrame(unique_extracted_skills_cv)
skills_cv = skills_cv.rename(columns={0: 'skill_description'})
skills_cv['Experience'] = np.random.randint(1, 6, size=len(skills_cv))
# for the purpose of clarity I have only choosen 7 skills from my resume
skills_cv = skills_cv.iloc[[1, 21, 23, 27, 36, 37, 65]]



#################################### build knowledge graph ################################

# pip install pyvis
# als Beispiel nehme ich jetzt mal doc1-doc5 um ein example project oder job-description zu simulieren
# und dann eben ncoh resume dazu machen
from pyvis.network import Network

job_net = Network(height='1000px', width='100%', bgcolor='#222222', font_color='white')
job_net.barnes_hut()
# create resume nodes
job_net.add_node('Resume', 'Resume', color='#dd4b39', title='doc1')
for skill in skills_cv.iterrows(): # iterate over each skill in order to create a node and connection to the resume node
    experience = skill[1]['Experience']
    skill = skill[1]['skill_description']
    job_net.add_node(skill, skill, title=skill)
    job_net.add_edge('Resume', skill, value=experience, color='#00ff1e', label='2')

# add job nodes
job_net.add_node('Job', 'Job', color='#dd4b39', title='doc1')
for skill in job_skills: # iterate over each skill in order to create a node and connection to the resume node
    job_net.add_node(skill, skill, title=skill)
    job_net.add_edge('Job', skill, value=1, color='#00ff1e', label='2')

job_net.show('job_graph.html', notebook=False)


######### Using Networkx ##########
import networkx as nx
import matplotlib.pyplot as plt

# Create an empty graph
graph = nx.Graph()

# Add the main 'Resume' node
graph.add_node('Resume', size=2000)

# List of skills with corresponding years of experience
skills = {'Python': 5, 'Java': 3, 'Machine Learning': 7, 'Data Analysis': 4}


# Add 'skill' nodes and edges connecting them to the 'Resume' node with weighted edges
for skill in skills_cv.iterrows(): # iterate over each skill in order to create a node and connection to the resume node
    experience = skill[1]['Experience']
    skill = skill[1]['skill_description']
    graph.add_node(skill)
    graph.add_edge('Resume', skill, weight=experience)


# Add the main 'Job_skills' node
graph.add_node('Job_skills', size=2000)

# List of skills for the 'Job_skills' node
job_skills = ['MLOps', 'AWS', 'NLP', 'ComputerVision', 'DeepReinforcementLearning']

# Add 'skill' nodes and edges connecting them to the 'Job_skills' node
for skill in job_skills:
    graph.add_node(skill)
    graph.add_edge('Job_skills', skill, weight=1)

weights = [graph.edges[edge]['weight'] for edge in graph.edges]
node_sizes = [graph.nodes[node]['size'] if ((node == 'Resume') | (node =='Job_skills')) else 100 for node in graph.nodes]
# Plot the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(graph)
nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue', font_size=12, font_color='black',
                 node_size=node_sizes,edge_color='grey', width=weights)
edge_labels = {(u, v): graph.edges[u, v]['weight'] for u, v in graph.edges}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

plt.axis('off')
plt.show()



