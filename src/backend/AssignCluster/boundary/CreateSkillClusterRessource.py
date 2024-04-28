import sys
sys.path.append('/Users/A200319269/PycharmProjects/job_resume_tool/')
from flask import Flask, jsonify
from src.backend.AssignCluster.control.SkillClusterCreatorBA import SkillClusterCreatorBA

app = Flask(__name__)

folder_path = '/Users/A200319269/PycharmProjects/job_resume_tool/Tech_data/ChatGPT_jira_stories'

@app.route('/cluster_dict', methods=['GET'])
def get_cluster_dict():
    skillClusterCreatorBA = SkillClusterCreatorBA(folder_path)
    cluster_dict = skillClusterCreatorBA.create_skill_cluster()
    return jsonify(cluster_dict)

if __name__ == '__main__':
    app.run(debug=True)