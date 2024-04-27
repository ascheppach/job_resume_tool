from src.backend.AssignCluster.control.helper_functions.helper_data_processing import open_folder
import json
import spacy
import openai
import ast

class SkillClusterAssistant:
    def __init__(self, skill_tree_dict_path, ner_model_path, resume_path):
        self.skill_tree_dict_path = skill_tree_dict_path
        self.ner_model_path = ner_model_path
        self.resume_path = resume_path

    def load_files(self):
        with open(self.skill_tree_dict_path, 'r') as json_file:
            skill_cluster_dict = json.load(json_file)
        with open(self.resume_path, 'r') as file:
            resume_content = file.read()
        return skill_cluster_dict, resume_content

    def load_model(self):
        ner_model = spacy.load(self.ner_model_path)
        return ner_model

    # erstmal aus dem dictionary lediglich die cluster rausextrahieren und eben nicht die einzelnen skills
    def extract_cluster_hierachy_levels(self, cluster_dict):
        self.clusters = list(cluster_dict.keys())

        sub_clusters = list(list(cluster_dict[str(dict)].keys()) for dict in cluster_dict)
        self.merged_sub_clusters = [item for sublist in sub_clusters for item in sublist]

        self.merged_skills = []
        for category in cluster_dict.values():
            for subcategory in category.values():
                self.merged_skills.extend(subcategory)
        self.all_hierachy_skills = self.clusters + self.merged_sub_clusters + self.merged_skills

    def build_llm_agent_for_existing_skills(merged_sub_clusters):
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

        return completions

    def create_skill_clusters(self):
        skill_cluster_dict, resume_content = self.load_files()
        self.extract_cluster_hierachy_levels(skill_cluster_dict)

        ner_model = self.load_model()

        extracted_skills_resume = []
        ner_resume_object = ner_model(resume_content)
        for entity in ner_resume_object.ents:
            extracted_skills_resume.append(entity.text)

        cluster_chain_cv = []
        no_skills = []
        for skill in extracted_skills_resume:
            print(skill)
            # skill = extracted_skills_cv[4]
            # skill = 'Machine Learning'
            # skill = 'ticket'
            # If skill from resume already exists in our current general skill_tree it can be directly assigned to current skill-tree
            if skill in self.all_hierachy_skills:
                print('Skill already an element in our current general skill-tree.')

            # If skill from resume does not exist in our current general skill_tree,
            # the skill-tree needs to get updated and extended
            else:
                final_question = cluster_question_2.format(skill, json.dumps(merged_sub_clusters))
                completions = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant for clustering und summarizing terms."},
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