import os
import spacy
import nltk
import re
from stop_word_list import *
import json
import openai
from openai import OpenAI

def open_folder(folder_path):
    all_files = []
    # Iterate over each file in the folder and append its content
    for filename in os.listdir(folder_path):
        # print(filename)
        if filename.endswith('.txt'):
            # print(filename)
            file_path = os.path.join(folder_path, filename)

            # Open the file and read its contents
            with open(file_path, 'r') as file:
                content = file.read()
                content = content.split('Title')
                all_files += [cont.replace("\nDescription: ", ". ") for cont in content]

    return all_files

nlp_ner = spacy.load("/Users/A200319269/PycharmProjects/job_resume_tool/ner_models/model-best")
regex = re.compile('[' + re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^`{|}~') + '\\r\\t\\n]')

def extract_skill_entities(all_files):
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
    return extracted_entities_all



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


def create_abbreviating_dictionary(list):
    mapping = {}

    for string in list:
        words = string.split()
        if len(words) > 1:
            abbreviation = ''.join(word[0] for word in words)
            mapping[abbreviation.lower()] = string
    mapping['aws'] = 'amazon web services'
    mapping['gke'] = 'google kubernetes engine'
    mapping['gcp'] = 'google cloud platform'
    return mapping


def transformAbbreviations(skill_list):
    mapping = {}
    updated_skill_list = []
    for doc in skill_list:
        # doc = 'aws-ec2'
        # str = 'aws'
        new_term = ''
        for i, str in enumerate(doc.split()):
            if i==0:
                new_term += mapping.get(str.lower(), str)
            else:
                new_term += ' ' + mapping.get(str.lower(), str)
        updated_skill_list.append(new_term)
    return updated_skill_list

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


def summarize_skill_terms(result_list):
    response_list = []
    for i, term_cluster in enumerate(result_list):
        print(i)
        if i > 20:
            continue
        try:
            if len(term_cluster) > 1:
                prompt = "What could be a summarizing term for the following word cluster: {}? Please only return the summarizing term and no additional text. If not found, return 'Not a skill'."
            else:
                prompt = "Is this term describing a technical skill: {}? If not, return 'Not a skill'"

            prompt = prompt.format(term_cluster)
            client = OpenAI()
            completions = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for clustering and summarizing terms."},
                    {"role": "user", "content": prompt}
                ]
            )
            message = completions.choices[0].message.content
            if message != 'Not a skill':
                response_list.append(message if len(term_cluster) > 1 else term_cluster[0])
        except openai.error.OpenAIError as e:
            print("OpenAI Server Error happened here.")
    return response_list


def create_skill_clusters(response_list):

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

    example_clusters_string = json.dumps(example_clusters)

    # create skill_cluster based on the summarized terms.
    summarize_term = "Please cluster following skills and also include different hierarchy levels or subclusters: {}."
    result_string = summarize_term.format(response_list)
    # create example clustering
    example_result_string = summarize_term.format(example_response_list)

    client = OpenAI()
    completions = client.chat.completions.create(
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

    return my_cluster_dict
