from src.backend.AssignCluster.control.SkillClusterCreatorBA import SkillClusterCreatorBA
from flask import Flask, jsonify
from src.backend.AssignCluster.control.helper_functions.helper_data_processing import open_folder, clean_data
from src.backend.NamedEntityRecognition.control.ExtractHelperBA import extract_skill_entities
from src.backend.AssignCluster.control.helper_functions.helper_bigrams import get_bigramms, get_trigramms
from src.backend.AssignCluster.control.helper_functions.helper_abbreviations import transformAbbreviations, create_abbreviating_dictionary
from src.backend.AssignCluster.control.helper_functions.helper_clustering import cluster_overlapping_strings
from collections import Counter
import json
import openai
from openai import OpenAI

app = Flask(__name__)

class SkillClusterCreatorBA:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    # Your existing methods here...

    @app.route('/cluster_dict', methods=['GET'])
    def get_cluster_dict(self):
        skill_cluster_creator = SkillClusterCreatorBA(self.folder_path)
        cluster_dict = skill_cluster_creator.create_skill_cluster()
        return jsonify(cluster_dict)

if __name__ == '__main__':
    folder_path = '/Users/A200319269/PycharmProjects/job_resume_tool/Tech_data/ChatGPT_jira_stories'

    app.run(debug=True)