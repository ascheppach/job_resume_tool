def create_abbreviating_dictionary(list):
    mapping = {}

    for string in list:
        words = string.split()
        if len(words) > 1:
            abbreviation = ''.join(word[0] for word in words)
            mapping[abbreviation.lower()] = string
    mapping['aws'] = 'amazon web services'
    mapping['gke'] = 'google kubernetes engine'
    mapping['gcp'] = 'google cloud platform'
    return mapping

def transformAbbreviations(skill_list):
    mapping = {}
    updated_skill_list = []
    for doc in skill_list:
        # doc = 'aws-ec2'
        # str = 'aws'
        new_term = ''
        for i, str in enumerate(doc.split()):
            if i==0:
                new_term += mapping.get(str.lower(), str)
            else:
                new_term += ' ' + mapping.get(str.lower(), str)
        updated_skill_list.append(new_term)
    return updated_skill_list