from src.backend.AssignCluster.control.helper_functions.helper_data_processing import open_folder, clean_data
from src.backend.NamedEntityRecognition.control.ExtractHelperBA import extract_skill_entities
from src.backend.AssignCluster.control.helper_functions.helper_bigrams import get_bigramms, get_trigramms
from src.backend.AssignCluster.control.helper_functions.helper_abbreviations import transformAbbreviations, create_abbreviating_dictionary
from src.backend.AssignCluster.control.helper_functions.helper_clustering import cluster_overlapping_strings
from collections import Counter

import json
import openai
from openai import OpenAI

class SkillClusterCreatorBA:
    def __init__(self, folder_path):
        self.folder_path = folder_path

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
                        {"role": "system",
                         "content": "You are a helpful assistant for clustering and summarizing terms."},
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

        example_response_list = ['EC2', 'S3', 'RDS', 'Lambda', 'Virtual Machines', 'Blob Storage', 'SQL Database',
                                 'Functions',
                                 'Compute Engine', 'Cloud Storage', 'Cloud SQL', 'Cloud Functions',
                                 'Jenkins', 'GitLab CI/CD', 'Travis CI', 'Kubernetes', 'Docker Swarm', 'Amazon ECS',
                                 'Terraform', 'AWS CloudFormation', 'Azure Resource Manager', 'NumPy', 'Pandas',
                                 'Scikit-learn',
                                 'React', 'Angular', 'Vue.js', 'SQL', 'GraphQL']

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

    def create_skill_cluster(self):
        all_files = open_folder(self.folder_path)
        clean_df = clean_data(all_files)
        bigrams = get_bigramms(clean_df)
        trigrams = get_trigramms(clean_df)
        abbreviation_dict = create_abbreviating_dictionary(trigrams)
        extracted_entities_all = extract_skill_entities(all_files)
        merged_list = [item for sublist in extracted_entities_all for item in sublist]
        transformed_skill_list = transformAbbreviations(merged_list)
        skill_frequency = Counter(transformed_skill_list)
        unique_skills = list(skill_frequency.keys())
        result_list = cluster_overlapping_strings(unique_skills)
        response_list = self.summarize_skill_terms(result_list)
        my_cluster_dict = self.create_skill_clusters(response_list)
        return my_cluster_dict