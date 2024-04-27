from openai import OpenAI

from src.backend.AssignCluster.control.helper_functions.helper_data_processing import open_folder
import json
import spacy
import openai
import ast

class AssignClusterBA:
    def __init__(self, example_messages_existing, example_messages_nonexisting, skill_tree_dict_path, ner_model_path, resume_path):
        self.example_messages_existing = example_messages_existing
        self.example_messages_nonexisting = example_messages_nonexisting
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

    def initialize_llm_agent_for_existing_skills(self):
        client = OpenAI()
        self.llm_existing = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=self.example_messages_existing
        )

    def initialize_llm_agent_for_nonexisting_skills(self):
        client = OpenAI()
        self.llm_nonexisting = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=self.example_messages_nonexisting
        )

    def create_skill_clusters(self):
        skill_cluster_dict, resume_content = self.load_files()
        self.extract_cluster_hierachy_levels(skill_cluster_dict)

        ner_model = self.load_model()

        extracted_skills_resume = []
        ner_resume_object = ner_model(resume_content)
        for entity in ner_resume_object.ents:
            extracted_skills_resume.append(entity.text)

        self.initialize_llm_agent_for_existing_skills()
        self.cluster_chain_resume = []
        no_skills = []
        for skill in extracted_skills_resume:
            print(skill)
            # skill = extracted_skills_cv[4]
            # skill = 'Machine Learning'
            # skill = 'ticket'
            # If skill from resume already exists in our current general skill_tree it can be directly assigned to current skill-tree
            if skill in self.all_hierachy_skills:
                print('Skill already an element in our current general skill-tree.')

                task_existing = ("Please assign the skill {} to a skillcluster from following list: {}. It can also be assigned to multiple clusters."
                .format(skill, json.dumps(self.merged_sub_clusters)))
                self.llm_existing.messages[-1]["content"] = task_existing
                completions = self.openAiClient.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    messages=self.llm_existing.messages
                )

            # If skill from resume does not exist in our current general skill_tree,
            # the skill-tree needs to get updated and extended
            else:
                task_nonexisting = ("Please assign the skill {} to a skillcluster from following list: {}. It can also be assigned to multiple clusters. If the skill is a synonym of a skillcluster from the list, then return the skillcluster from the list. If the term is not a skill then return 'Not Provided'."
                                    .format(skill, json.dumps(self.merged_sub_clusters)))
                self.llm_nonexisting.messages[-1]["content"] = task_nonexisting
                completions = self.openAiClient.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    messages=self.llm_nonexisting.messages
                )

            message = completions.choices[0].message.content
            if message == 'Not Provided':
                no_skills.append(skill)
            else:
                cluster_chain = ast.literal_eval(message)
                self.cluster_chain_resume.append(cluster_chain)


    def store_cluster_dict(self, file_name):
        with open(file_name, 'w') as json_file:
            json.dump(self.cluster_chain_resume, json_file)

    def load_cluster_dict (self, file_name):
        with open(file_name, 'r') as json_file:
            self.cluster_dict = json.load(json_file)
            return self.cluster_dict
    def clean_cluster(self):

        data = self.load_cluster_dict('scheppach_skill_new.json')
        new_data = []
        # new_data ist quasi einfach nur geflattened version vom dictionary
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
            for item in group:  # iteriert über: [{'Data Analysis': {'Statistical Analysis': []}}, {'Data Analysis': {'Statistical Analysis': []}}, {'Data Analysis': {'Data Scientist': []}}]
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