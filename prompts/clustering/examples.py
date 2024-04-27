cluster_question_1 = "Please assign the skill {} to a skillcluster from following list: {}. It can also be assigned to multiple clusters."

example_question_1 = cluster_question_1.format('Machine Learning', json.dumps(merged_sub_clusters))
example_answer_1 = "[{'Machine Learning': []}]"
# Potentiell wäre das hier möglich aber lieber immer nur Überbegriff bei solchen Fällen: Cluster 'Machine Learning' ; Subcluster 'Big Data': {'Machine Learning'} ; Subcluster 'Data Analysis': {'Machine Learning'}
# d.h. als regel soll er mir nur das oberste zurückgeben

# Case es gibt ihn und es ist ein Skill und kommt mehrmals vor in verschiedenen Cluster-Subcluster chains
example_question_2 = cluster_question_1.format('Object Detection', json.dumps(merged_sub_clusters))
example_answer_2 = "[{'Machine Learning': {'Computer Vision': ['Object Detection']}}, {'Artificial Intelligence': {'Computer Vision': ['Object Detection']}}]"

# Case es ist ein Subcluster und wir geben 2er Chain zurück
example_question_3 = cluster_question_1.format('Deep Learning', json.dumps(merged_sub_clusters))
example_answer_3 = "[{'Machine Learning': {'Deep Learning': []}}]"

example_question_4 = cluster_question_1.format('Ruby', json.dumps(merged_sub_clusters))
example_answer_4 = "[{'Software Development': {'Backend Development': ['Ruby']}}]"

example_question_synonym_1 = cluster_question_1.format('Azure Services', json.dumps(merged_sub_clusters))
example_answer_synonym_1 = "[{'Cloud Services': {'Azure': []}}]"

example_question_synonym_2 = cluster_question_1.format('NLP algorithms', json.dumps(merged_sub_clusters))
example_answer_synonym_2 = "[{'Artificial Intelligence': {'Natural Language Processing': []}},{'Machine Learning': {'Natural Language Processing': []}}]"

# abbreviation examples
example_question_abbreviation_1 = cluster_question_1.format('Amazon Web Services',
                                                            json.dumps(merged_sub_clusters))
example_answer_abbreviation_1 = "[{'Cloud Services': {'AWS': []}}]"

example_question_abbreviation_2 = cluster_question_1.format('NLP', json.dumps(merged_sub_clusters))
example_answer_abbreviation_2 = "[{'Artificial Intelligence': {'Natural Language Processing': []},{'Machine Learning': {'Natural Language Processing': []}]"
