from flask import Flask, request
import PyPDF2

app = Flask(__name__)

import PyPDF2

@app.route('/searchApplicants', methods=['POST'])
def search_applicants():
    file_contents = []
    for key, value in request.files.items():
        print(value)
        #print(key)
        if key.startswith('file'):
            pdf_file = value.stream
            reader = PyPDF2.PdfFileReader(pdf_file)
            text = ''
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extract_text()
            file_contents.append(text)
    print(file_contents)
    return 'Data received and processed successfully'


if __name__ == '__main__':
    app.run()
