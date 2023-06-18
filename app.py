from flask import Flask, request, jsonify
import json

# logging.basicConfig(level=logging.INFO, filename='log.txt')

app = Flask(__name__)

skill_list = []
@app.route('/skillList', methods=['POST'])
def add_skill():
    print('hello')
    new_skill = json.loads(request.form.get('skillList'))
    print(new_skill)
    skill_list.append(new_skill)
    return jsonify(new_skill)


important_skills = []
@app.route('/importantSkills', methods=['POST'])
def receive_important_skills():
    print('hello')
    importantskills = json.loads(request.form.get('importantSkills'))
    # important_skills.append(new_skills)
    print(importantskills)
    # print(new_skill)
    return jsonify({'message': 'Important skills received successfully'})


@app.route('/searchApplicants', methods=['POST'])
def search_applicants():
    # Get the skillList and importantSkills from the request
    skill_list = request.form.get('skillList')
    important_skills = request.form.get('importantSkills')

    # Read the file contents
    file_contents = []

    for file_key in request.files:
        file = request.files[file_key]
        file_content = file.read()
        print(file_content)
        file_contents.append(file_content.decode())

    print(file_contents[0])
    # Example: Return a response with the processed data
    response_data = {
        'skillList': skill_list,
        'importantSkills': important_skills,
        'fileContents': file_contents,
    }

    return jsonify(response_data)



if __name__ == '__main__':
    app.run()
