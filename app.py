from flask import Flask, request, jsonify
import logging
# logging.basicConfig(level=logging.INFO, filename='log.txt')

app = Flask(__name__)

skill_list = []
@app.route('/skillList', methods=['POST'])
def add_skill():
    new_skill = request.get_json()
    skill_list.append(new_skill)
    logging.info('New skill added: %s', new_skill)
    return new_skill


if __name__ == '__main__':
    app.run()
