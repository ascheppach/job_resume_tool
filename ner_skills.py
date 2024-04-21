
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import PyPDF2


nlp = spacy.blank("en")  # load a new spacy model
db = DocBin()  # create a DocBin object

import json
import os


folder_path = 'C:/Users/SEPA/topic_modeling/labeled_entities'  # Replace with the path to your folder

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    f = open(folder_path + '/' + filename)
    TRAIN_DATA = json.load(f)

    for text, annot in tqdm(TRAIN_DATA['annotations']): # text ist eben text, annot sind die gelabelten annotations
        # print(text) # text
        # print(annot) # die annotierten entities
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]: # ents sind einfach nur die beiden wörter die er sich aus start und end zusammenbaut
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

db.to_disk("./training_data_skills.spacy")  # save the docbin object


# In der Console (um config Datei zum trainieren zu erstellen)
# python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency

# In der Console (um zu trainieren)
# nach paths.train immer training_data und nach paths.dev immer validation_data
# python -m spacy train config.cfg --output ./ --paths.train ./training_data_skills.spacy --paths.dev ./training_data_skills.spacy

# import model-best
nlp_ner = spacy.load("C:/Users/SEPA/topic_modeling/model-best")

doc = nlp_ner('''I have several years of experience with NLP and MLOps. Here I implemented a Text Classification Algorithm with BERT Algorithm. Moreover I have worked with AWS, Kubernetes and Docker.''')
#spacy.displacy.render(doc, style="ent", jupyter=True)
extracted_skills = []
for ent in doc.ents:
    extracted_skills.append(ent.text)



def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        text = ''
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()

        return text

pdf_file_path = 'C:/Users/SEPA/lanchain_ir2/CV_Scheppach_text.txt'
pdf_text = read_pdf(pdf_file_path)

file_path = 'C:/Users/SEPA/lanchain_ir2/CV_Scheppach_text.txt'

with open(file_path, 'r') as file:
    content = file.read()

doc = nlp_ner(content)

extracted_skills_cv = []
for ent in doc.ents:
    extracted_skills_cv.append(ent.text)

