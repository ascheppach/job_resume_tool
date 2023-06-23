from flask import Flask, request, send_file
import PyPDF2
import json
from job_resume_matching_phraseMatcher import compare_resumes_to_job, sort_applicants, create_plot_resume_to_job

app = Flask(__name__)


@app.route('/searchApplicants', methods=['POST'])
def search_applicants():
    skill_list = json.loads(request.form.get('skillList'))
    important_skills = json.loads(request.form.get('importantSkills'))
    #print(skill_list)
    #print(important_skills)

    file_contents = []
    for key, value in request.files.items():
        if key.startswith('file'):
            pdf_file = value.stream
            #print(pdf_file)
            reader = PyPDF2.PdfFileReader(pdf_file)
            #print(reader)
            text = ''
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                #print(page)
                #print(page.extract_text())
                text += page.extract_text()
            file_contents.append(text)
    #print(file_contents)

    new_data = compare_resumes_to_job(file_contents, skill_list)
    new_data = sort_applicants(new_data, important_skills)
    create_plot_resume_to_job(new_data)

    return 'Data received and processed successfully'

@app.route('/get_skillcluster_image', methods=['GET'])
def get_skillcluster_image():
    # Replace 'path/to/your/image.png' with the actual path to your stored PNG file
    filename = 'skill_plot.png'

    try:
        return send_file(filename, mimetype='image/png')
    except Exception as e:
        return str(e), 404


if __name__ == '__main__':
    app.run()
