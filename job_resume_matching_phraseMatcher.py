# Resume Phrase Matcher code


# importing all required libraries

import PyPDF2
import os
from io import StringIO
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher
from compare_applicants import get_applicantName

import matplotlib.pyplot as plt
import numpy as np

# Function to read resumes from the folder one by one
import pandas as pd

# create DataFrame

def compare_resumes_to_job(resumes_path, keyword_path):

    onlyfiles = [os.path.join(resumes_path, f) for f in os.listdir(resumes_path) if os.path.isfile(os.path.join(resumes_path, f))]

    def pdfextract(file):
        fileReader = PyPDF2.PdfFileReader(open(file, 'rb'))
        countpage = fileReader.getNumPages()
        count = 0
        text = []
        while count < countpage:
            pageObj = fileReader.getPage(count)
            count += 1
            t = pageObj.extractText()
            print(t)
            text.append(t)
        return text


    # function that does phrase matching and builds a candidate profile
    def create_profile(file):
        # file = onlyfiles[0]
        text = pdfextract(file)
        text = str(text)
        text = text.replace("\\n", " ")
        text = text.lower()
        keyword_dict = pd.read_excel(keyword_path);
        # below is the csv where we have all the keywords, you can customize your own
        keyword_list = []
        for subject in keyword_dict.columns:
            keyword_list.append([nlp(text) for text in keyword_dict[subject].dropna(axis=0)])

        matcher = PhraseMatcher(nlp.vocab)
        for i in range(len(keyword_list)):
            matcher.add(keyword_dict.columns[i], None, *keyword_list[i])

        doc = nlp(text)

        d = []
        matches = matcher(doc)
        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
            span = doc[start: end].text  # get the matched slice of the doc
            d.append((rule_id.replace(' ', ''), span.replace(' ', '')))
        keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i, j in Counter(d).items())

        ## convertimg string of keywords to dataframe
        df = pd.read_csv(StringIO(keywords), names=['Keywords_List'])
        df_final = pd.DataFrame(columns = ['subject', 'keyword', 'count'])
        for row in range(len(df)): # iterate over keywords of this resume
            # row = 0
            row = df.Keywords_List[row].split()
            subject = row[0].replace('(', '').replace(')', '')
            keyword = row[1].replace('(', '').replace(')', '')
            count = row[2].replace('(', '').replace(')', '')
            new_row = pd.DataFrame({'subject': subject, 'keyword': keyword, 'count': count}, index=[0])
            df_final = pd.concat([new_row, df_final.loc[:]]).reset_index(drop=True)

        base = os.path.basename(file)
        filename = os.path.splitext(base)[0]

        # name = filename.split('_')
        # name2 = name[1]
        name = get_applicantName(base)
        name = name.split(' ')[-1]
        ## converting str to dataframe
        name3 = pd.read_csv(StringIO(name), names=['Candidate Name'])

        dataf = pd.concat([name3['Candidate Name'], df_final['subject'], df_final['keyword'], df_final['count']], axis=1)
        dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace=True)

        return (dataf)

    # function to read resume ends
    i = 0
    while i < len(onlyfiles):
        file = onlyfiles[i]
        dat = create_profile(file)

        if i > 0:
            frames = [final_data, dat]
            final_data = pd.concat(frames)
        else:
            final_data = dat
        i += 1
    # function ends

    # code to execute/call the above functions

    # code to count words under each category and visulaize it through Matplotlib
    final_database = final_data
    final_database2 = final_database['keyword'].groupby([final_database['Candidate Name'], final_database['subject']]).count().unstack()
    final_database2.reset_index(inplace=True)
    final_database2.fillna(0, inplace=True)
    new_data = final_database2.iloc[:, 1:]
    new_data.index = final_database2['Candidate Name']

    for index, row in new_data.iterrows():
        row_sum = new_data.sum(axis=1) # idx0 16,13 ; idx1 16,13
        new_data.loc[index] = row/row_sum[index] # idx0 2/16, 3/16 ; idx1 2/13, 4/13
        #print(new_data.loc[index])

    return new_data



def create_plot_resume_to_job(data):
    new_data = data
    # bars an richtiger stelle
    # benamung der bars an richtiger stelle
    categories = list(new_data.index)

    values = []
    for column_index in range(len(new_data.columns)):
        column_values = list(new_data.iloc[:, column_index])
        values.append(column_values)

    # haben 2x8 liste, brauchen aber 8x2 liste
    num_categories = len(categories)
    num_values = len(values)

    # Calculate the width of each bar
    bar_width = 0.0875 # 0.7 / num_values

    # Plotting the stacked bar graph
    plt.figure(figsize=(10, 9))
    bars = []
    for i in range(num_values): # er iteriert über die subject, brauchen also eigentl für jedes subject ein x pair (für jede
        # category eins): genau das wird auch gemacht, wir haben am Ende 8 bars und jede bar hat 2 rectangle (und dieses
        # rectangle hat wiederrum xy pair)
        # i=3
        if i == 0:
            # er benammt es deshalb mit Value0, dann Value1 usw.
            bars.append(plt.bar(categories, values[i], width=bar_width, label='Value {}'.format(i+1), bottom=np.sum((values[:i]), axis=0)))
        else:
            bars.append(plt.bar(categories, values[i], width=bar_width, label='Value {}'.format(i+1), bottom=np.sum((values[:i]), axis=0)))

    x_space = 0.1
    end = (x_space * num_categories)-x_space
    x_pos = np.arange(0, end, x_space)
    # hier wird definiert wo die bars positioniert werden
    for i in range(len(bars)): # for i 1:8
        for s in range(num_categories): # for i 1:3
            bars[i][s].set_x(x_pos[s])


    # Adding labels and legend
    plt.xlabel('Applicants')
    plt.ylabel('Score')
    plt.title('Comparison of applicants')

    legend_labels = list(new_data.columns)
    plt.legend(legend_labels, loc='upper right')
    plt.xticks(range(num_categories), categories)

    # jetzt wird definiert an welcher Stelle die Prozentzahlen geschrieben werden (welche x und y position)!!!
    for i, bar in enumerate(bars):
        # i=0,1,...,7 -> die 8 subjects
        idx = 0
        for rect in bar:
            # und für jedes subject gibt es zwei rect objecte (was auch Sinn macht, weil für jedes subject gibt es ja auch
            # 2 applicants/categories):
            # - dieses rect object hat height, was einfach nur diese prozentangabe ist
            # - width definiert einfach nur wie breit die bars sind
            # - y gibt an von wo aus der nächste block anfängt auf der y-achse
            # - x gibt an auf welcher position an der x-achse die bars positioniert werden (deshalb gibt es nur 2 stück und
            #   die sind auch immer gleich: 0.95625 , -0.04375
            height = round(rect.get_height(), 2)
            # round(bar[1].get_height(), 2)

            width = rect.get_width()
            rect.set_x(x_pos[idx])
            x = rect.get_x()
            # print(x)
            y = rect.get_y()
            # jetzt wird definiert an welcher Stelle die Prozentzahlen geschrieben werden (welche x und y position)
            plt.text(x + width / 2, y + height / 2, '{}%'.format(round(height*100),2), ha='center', va='center')
            idx += 1
    plt.xticks(np.array(x_pos), categories, rotation='vertical')
    plt.savefig('skill_plot.png')
    # plt.show()



# data = new_data
def sort_applicants(data, important_skills):

    # Reorder the columns
    sorted_columns = important_skills + [col for col in data.columns if col not in important_skills]
    df = data[sorted_columns]

    scores, indices = [], []
    for index, row in df.iterrows():
        scores.append(np.sum(row[important_skills]))
        indices.append(index)
    summary_df = pd.DataFrame({ 'Indices': indices, 'Scores': scores})
    df_sorted = summary_df.sort_values(by='Scores', ascending=False)
    desired_order = list(df_sorted['Indices'])
    # Reorder the DataFrame based on the desired order
    df_ordered = df.reindex(desired_order)
    return df_ordered


#resume_directory = 'C:/Users/SEPA/lanchain_ir2/Resume_data_pdf'  # enter your path here where you saved the resumes
#skill_file = 'C:/Users/SEPA/lanchain_ir2/job_demand.xlsx'
#new_data = compare_resumes_to_job(resume_directory, skill_file)

#important_skills = ['NLP', 'MLOps']
#new_data = sort_applicants(new_data, important_skills)
#create_plot_resume_to_job(new_data)