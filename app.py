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
    new_skills = request.get_json()
    important_skills.extend(new_skills)
    return jsonify({'message': 'Important skills received successfully'})


if __name__ == '__main__':
    app.run()
