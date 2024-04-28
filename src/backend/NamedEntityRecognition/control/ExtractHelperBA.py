import spacy
import re

class SkillEntityExtractor:
    def __init__(self, model_path):
        self.nlp_ner = spacy.load(model_path)
        self.regex = re.compile('[' + re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^`{|}~') + '\\r\\t\\n]')

    def extract_skill_entities(self, all_files):
        extracted_entities_all = []
        for sample in all_files:
            skill_entities = self.nlp_ner(sample)
            extracted_skills = []
            for ent in skill_entities.ents:
                extracted_skills.append(ent.text)
            if extracted_skills:
                extracted_skills = [element.lower() for element in extracted_skills]
                extracted_skills = [self.regex.sub(" ", element) for element in extracted_skills]
                extracted_entities_all.append(extracted_skills)
        return extracted_entities_all