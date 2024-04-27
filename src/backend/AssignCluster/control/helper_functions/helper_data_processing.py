from stop_word_list import *
import pandas as pd
import os

def open_folder(folder_path):
    all_files = []
    # Iterate over each file in the folder and append its content
    for filename in os.listdir(folder_path):
        # print(filename)
        if filename.endswith('.txt'):
            # print(filename)
            file_path = os.path.join(folder_path, filename)

            # Open the file and read its contents
            with open(file_path, 'r') as file:
                content = file.read()
                content = content.split('Title')
                all_files += [cont.replace("\nDescription: ", ". ") for cont in content]

    return all_files

def clean_data(data):
    df = pd.DataFrame(data)
    df = df.rename(columns={0: 'skill_description'})
    df = df[df['skill_description'] != '']
    for index, row in df.iterrows():
        # print(df.iloc[index,0])
        row['skill_description'] = row['skill_description'].replace("\n", " ")
        row['skill_description'] = row['skill_description'].replace("Description:", "")
    clean_df = clean_skills(df, 'skill_description')
    return clean_df