import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.blank("en")  # load a new spacy model
db = DocBin()

import json
import os
import ast


folder_path = 'C:/Users/SEPA/lanchain_ir2/labeled_entities_time_skill'  # Replace with the path to your folder

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    f = open(folder_path + '/' + filename)
    TRAIN_DATA = json.load(f)

    for text, annot in tqdm(TRAIN_DATA['annotations']): # text ist eben text, annot sind die gelabelten annotations
        # print(text) # text
        # print(annot) # die annotierten entities
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]: # ents sind einfach nur die beiden wörter die er sich aus start und end zusammenbaut
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

db.to_disk("./training_data_skills_time.spacy")

# python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency
# python -m spacy train config.cfg --output ./ --paths.train ./training_data_skills_time.spacy --paths.dev ./training_data_skills_time.spacy



############################################################ Ab hier ######################################################
# get NER model which predicts time and
nlp_ner = spacy.load("C:/Users/SEPA/lanchain_ir2/model-best")
file_path = 'C:/Users/SEPA/lanchain_ir2/CV_Scheppach_text.txt'
with open(file_path, 'r') as file:
    content = file.read()
doc = nlp_ner(content)
extracted_skills_cv = []
for ent in doc.ents:
    extracted_skills_cv.append(ent.text)
print(extracted_skills_cv)

import openai
filename = 'C:/Users/SEPA/lanchain_ir2/my_cluster_dict_new.json'
with open(filename, 'r') as json_file:
    my_cluster_dict = json.load(json_file)
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
example_answer_1 = "['Machine Learning']"
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
example_answer_6 = "['Cloud services']"

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

time_question = "Is following term describing time: {}? Only give response 'yes' or 'no'."
time_difference_question = "How many months does this time range include: {}?"


all_dicts = []
no_skills = []
for i, skill in enumerate(extracted_skills_cv):
    print(skill)
    # skill = extracted_skills_cv[4]
    # skill = 'Machine Learning'
    # skill = 'ticket'
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for defining weather a term is time or not."},
            {"role": "user", "content": "Is following term describing time: 04/2020-01/2022? Only give response 'yes' or 'no'."},
            {"role": "assistant", "content": 'yes'},
            {"role": "user", "content": "Is following term describing time: 'Machine Learning'? Only give response 'yes' or 'no'."},
            {"role": "assistant", "content": 'no'},

            {"role": "user", "content": time_question.format(skill)}
        ]
    )
    message = completions.choices[0].message.content

    print(message)
    if message == 'yes':
        print('Time!')

        time = skill
        completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant for calculation the time difference in months."},
                {"role": "user",
                 "content": "How many months does this time range include: 04/2020-01/2022?"},
                {"role": "assistant", "content": '22'},
                {"role": "user", "content": time_difference_question.format(time)}
            ]
        )
        message = completions.choices[0].message.content
        time = message
        if i > 0:
            all_dicts.append(new_dict)
        # time = '12/2019-05/2020'
        new_dict = {time: []}
    else:
        # skill = 'ML'
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
            new_dict[time].append(cluster_chain)

'05/2020-08/2020'

