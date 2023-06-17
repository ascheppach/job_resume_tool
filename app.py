from flask import Flask, request
from flask.helpers import send_from_directory, send_file
# from flask_cors import CORS, cross_origin
from job_resume_matching_phraseMatcher import compare_resumes_to_job, sort_applicants, create_plot_resume_to_job

app = Flask(__name__)#, static_folder='my-app/build', static_url_path='')
# CORS(app)

@app.route('/api', methods=['GET'])
# @cross_origin()
def index():
    return {
        "tutorial": "Flask React Heroku"
    }

@app.route('/compare_Corresponding skill catalogue', methods=['GET'])
def get_skill_plot():
    resume_directory = 'C:/Users/SEPA/lanchain_ir2/Resume_data_pdf'  # enter your path here where you saved the resumes
    skill_file = 'C:/Users/SEPA/lanchain_ir2/job_demand.xlsx'
    skills = ['NLP', 'MLOps']

    new_data = compare_resumes_to_job(resume_directory, skill_file)
    new_data = sort_applicants(new_data, skills)
    create_plot_resume_to_job(new_data)

    return send_file('skill_plot.png', mimetype='image/png')

skill_list = []
@app.route('/skillList', methods=['POST'])
def add_skill():
    new_skill = request.json
    skill_list.append(new_skill)
    return 'Skill added successfully', 200

@app.route('/skillList', methods=['GET'])
def get_skill_list():
    return {'skillList': skill_list}, 200

#@app.route('/')
#def serve():
#    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run()