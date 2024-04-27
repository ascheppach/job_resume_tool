import spacy
import re

nlp_ner = spacy.load("/Users/A200319269/PycharmProjects/job_resume_tool/ner_models/model-best")
regex = re.compile('[' + re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^`{|}~') + '\\r\\t\\n]')

def extract_skill_entities(all_files):
    extracted_entities_all = []
    for sample in all_files: # list und jedes element ist
        # sample = all_files[3]
        doc = nlp_ner(sample)
        extracted_skills = []
        for ent in doc.ents:
            extracted_skills.append(ent.text)
        if extracted_skills: # Code to be executed if the list is not empty
            #extracted_skills = [element.replace(' ', '_').replace("-", "_") for element in extracted_skills] # einzelne listenelemente mit "_"
            #extracted_entities_all.append(' '.join(extracted_skills)) # wird gejoined
            # extracted_skills = extracted_skills.lower()
            # remove punctuation: entferne kommas, slash zeichen usw.
            # df[col_name] = df[col_name].map(punc_skill)
            extracted_skills = [element.lower() for element in extracted_skills]
            extracted_skills = [regex.sub(" ", element) for element in extracted_skills]
            extracted_entities_all.append(extracted_skills)
        else:
            continue
    return extracted_entities_all