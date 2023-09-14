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

    # keyword_path = 'C:/Users/SEPA/lanchain_ir2/job_demand.xlsx'
    # onlyfiles = [os.path.join(resumes_path, f) for f in os.listdir(resumes_path) if os.path.isfile(os.path.join(resumes_path, f))]
    # test = pd.Series(keyword_path[0]['skills'])
    # testold = keyword_dict['Deep Learning']

    columns = []
    num_rows = []
    for values in keyword_path:
        columns.append(values['skill'])
        num_rows.append(len(values['skills']))
    max_row = max(num_rows)

    df = pd.DataFrame([], columns=columns, index=range(max_row))

    for cnt, column_name in enumerate(df.columns):
        skill_cluster = []
        for word in keyword_path[cnt]['skills']:
            skill_cluster.append(word.lower())
        df[column_name] = pd.Series(skill_cluster)


    # function that does phrase matching and builds a candidate profile
    def create_profile(text, df):
        ## es hackt akguell bei matches und zwar nicht wegen text, sondern wegen keyword_dict.
        # text = resumes_path[0]

        # onlyfiles = [os.path.join(resume_directory, f) for f in os.listdir(resume_directory) if os.path.isfile(os.path.join(resume_directory, f))]
        # file = onlyfiles[0]
        # text = pdfextract(file)

        text = str(text)
        text = text.replace("\\n", " ")
        text = text.replace("\n", " ")
        text = text.replace("PyT orch", "PyTorch")

        text = text.lower()
        keyword_dict = df
        # testobj = df['Deep Learning'][0] # str

        # keyword_dict = pd.read_excel(keyword_path) dataframe 9,7
        # below is the csv where we have all the keywords, you can customize your own
        keyword_list = []
        # subject = 'Deep Learning'
        # test = keyword_dict[subject] Series mit den keywords
        for subject in keyword_dict.columns:
            #for text in keyword_dict[subject].dropna(axis=0):
             #   print(text) # PyTorch , Tensorflow, Python, R
            keyword_list.append([nlp(text) for text in keyword_dict[subject].dropna(axis=0)]) # [[neural network, pytorch, tensorflow, deep learning], [bert, nlp, langchain], [spark, scala], [pata preprocessing, sql, spark, scala, etl ], [prediction, predictive modeling, random forest, python, r, classification, regression], [aws, azure, gcp, sagemaker], [aws, kubernetes, cloud, shell, software development, docker, container, sagemaker, kubernetes]]


        matcher = PhraseMatcher(nlp.vocab) # <spacy.matcher.phrasematcher.PhraseMatcher object at 0x0000018AA1583920>
        for i in range(len(keyword_list)): # über jedes subject iterieren i=0
            matcher.add(keyword_dict.columns[i], None, *keyword_list[i]) # 'Deep Learning', None, [neural network, pytorch, tensorflow, deep learning]

        doc = nlp(text)

        d = []
        matches = matcher(doc) # [(17857678330435779591, 48, 50), (16378066519788692076, 56, 57) ....]
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

        # name = filename.split('_')
        # name2 = name[1]
        name = get_applicantName(text)
        name = name.split(' ')[-1]
        ## converting str to dataframe
        name3 = pd.read_csv(StringIO(name), names=['Candidate Name'])

        dataf = pd.concat([name3['Candidate Name'], df_final['subject'], df_final['keyword'], df_final['count']], axis=1)
        dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace=True)

        return (dataf)

    # function to read resume ends
    i = 0
    while i < len(resumes_path):
        file = resumes_path[i]
        dat = create_profile(file, df)

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
    # final_database2 = final_database['keyword'].groupby([final_database['Candidate Name'], final_database['subject']]).count()

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
def sort_applicants(data, skills):

    # need skillcolumns + all the other columns
    for i, skill in enumerate(skills):
        print(skill['text'].replace(" ", ""))
        skills[i] = skill['text'].replace(" ", "")

    sorted_columns = skills + [col for col in data.columns if col not in skills]
    df = data[sorted_columns]

    scores, indices = [], []
    for index, row in df.iterrows():
        scores.append(np.sum(row[skills]))
        indices.append(index)
    summary_df = pd.DataFrame({ 'Indices': indices, 'Scores': scores})
    df_sorted = summary_df.sort_values(by='Scores', ascending=False)
    desired_order = list(df_sorted['Indices'])
    # Reorder the DataFrame based on the desired order
    df_ordered = df.reindex(desired_order)
    return df_ordered


# resumes_path = [
#        'Curriculum V itae\nPersonal details\nName\nAmadeu Manfred Scheppach\nAddress\nSchleifweg 10A, 86405 Meitingen\nContact\nscheppachamadeu@yahoo.com\nMobile\n+4917\n9 4320800\nEducation\n04/2020-01/2022\nMaster Statistics (LMU)\n- \n⌀\n 1.68 (GP A\n3.3)\nᐨ\nFocus on Machine Learning, Deep Learning and AutoML \nᐨ\nProjects with Python, R, and Matlab\nWork Experience\nSince 04/2022\nJunior Data Scientist, Cognizant Mobility \nᐨ\nHere I am working in three dif ferent positions/teams, namely data science, software\ndevelopment and operations.\nData Science \nᐨ\nWorking on an NLP  project where we are developing an MLOps pipeline for ticket\nclassification \nᐨ\nUsing dif ferent NLP  algorithms and also a variety of AWS services, such as Lambda,\nSagemaker , EC2, S3, ECR, as well as Step Functions \nᐨ\nInstall, build and run docker containers with AWS\nSoftware Development \nᐨ\nWorking on the BMW eSIM backend system \nᐨ\nSolving defects as well as implementing new features with Java \nᐨ\nFurther skills: Kubernetes, Cloud (A WS), CI/CD, DevOps, Git W orkflows\nOperations \nᐨ\nSolving incidents by using SQL, PL/pgSQL, as well as AWS Services\n05/2021-1 1/2021\nMaster student, Helmholtz-Zentrum\nfür Infektionsforschung\n- Grade 1.3 (GP A \n3.7)\nᐨ\nImplemented new algorithms (based on popular NAS methods for image classification\nand NLP) in the field of bioinformatics to find the optimal Deep Learning architecture in\nan automated way (using Python and PyT orch) \nᐨ\nBuilt novel NAS methods (OSP-NAS, CWP-DAR TS, DEP-DAR TS) \nᐨ\nWe aim to publish part of the work at the upcoming AutoML  conference\n12/2018-06/2020\nWorking student Data Analytics, Interhyp\nAG\nᐨ\nBuilt predictive models to identify error-sources in the tracking system (using R) \nᐨ\nCustomer Journey Analysis (using Tableau)\nFurther programming skills & projects\n04/2021-08/2021\nApplied Deep Learning with T ensorFlow\nand PyT orch\n- Grade 1.0 (GP A 4.0)\nᐨ\nImplemented the SCINet architecture, based on the paper „T ime Series is a Special\nSequence: Forecasting with Sample Convolution and Interaction“ (using Python and\nPyTorch)\n08/2020\nHackathon Fraunhofer Institute\nᐨ\nPattern recognition for sensor data (time series clustering) \nᐨ\nImplemented Non-parametric Hidden Semi-Markov Models (with physmm library in\nPython)\n05/2020-08/2020\nReinforcement Learning\n- Grade 1.3\n(GPA 3.7)\nᐨ\nImplemented the Deep Reinforcement Learning algorithm PPO (using Python and\nOpenAI environment)\n12/2019-05/2020\nBayesian Optimization for Material\nScience\n- Grade 1.3 (GP A 3.7)\nᐨ\nBuilt the package „EBO“, designed for material science (using R) \nᐨ\nEnables chemists and researches a better understanding of their optimization and\nsimulation: tune the hyperparameters of optimization algorithms and understand the\neffect of its hyperparameters with contour plots and ablation analysis\nSkills and Interests\nProgramming\nPython, Java, R, Tableau, Matlab\nFurther skills\nCloud (A WS), MLOps, SQL, Deep Learning, PyT orch, Linux, NAS, AutoML, CI/CD, DevOps\nLanguages\nGerman (native), English (fluent) , Portuguese (fluent), French (intermediate)\nInterests\nMachine Learning, Chess, Surfing, Skiing',
#        "1 of 2 Juan Jose Carin  \nData Scientist   \nMountain View , CA 94041  \n 650-336-4590  | juanjose.carin@gmail.com  \n linkedin.com/in/juanjosecarin  | juanjocarin.github.io  \n \nProfessional  Profile  \nPassionate abo ut data analysis and experiments, mainly focused on user  behavior, experience , and engagement , with a solid \nbackground in data science and statistics, and extensive experience using data insights to drive business growth.  \nEducation\n2016  University  of California,  Berkeley  Master  of Information  and Data  Science  GPA:  3.93\n \n \n Relevant  courses : \n• Machine  Learning  \n• Machine  Learning  at Scale  \n• Storing  and Retrieving  Data  • Field  Experiments  \n• Applied  Regression  and Time  Series  \nAnalysis  \n• Exploring  and Analyzing  Data  • Data  Visualization  and \nCommunication  \n• Research  Design  and Applications  for \nData  Analysis  \n2014 Universidad  Politécnica  de Madrid  M.S.  in Statistical  and Computational  Information  Processing  GPA:  3.69\n \n \n Relevant  courses :  \n• Data  Mining  \n• Multivariate  Analysis  \n• Time  Series  • Neural  Networks  and Statistical  \nLearning  \n• Regression  and Prediction  Methods  \n• Optimization  Techniques  • Monte  Carlo  Techniques  \n• Numerical  Methods  in Finance  \n• Stochastic  Models  in Finance  \n• Bayesian  Networks\n2005  Universidad  Politécnica  de Madrid  M.S.  in Telecommunication  Engineering  GPA:  3.03\nFocus  Area:  Radio  communication  systems  (radar  and mobile).  \nFellowship:  First  year  at University,  due to Honors  obtained  last year  at high  school.  \nSkills  \n Programming  / Statistics  Big Data  Visual ization  Others  \nProficient:  R, Python , SQL Hadoop , Hive , MrJob  Tableau  Git, AWS  \nIntermediate:  SPSS , SAS, Matlab  Spark , Storm   Bash  \nBasic:  EViews , Demetra+   D3.js Gephi , Neo4j , QGIS  \nExperience  \nDATA SCIENCE  \nJan. 2016  – Mar.  2016 Data  Scientist  \n CONENTO   Madrid,  Spain  (working  remotely)  \n• Designed  and implemented  the ETL pipeline  for a predictive  model  of traffic  on the main  roads  in \neastern  Spain  (a project  for the Spanish  government) . \n• Automated  scripts  in R to extract,  transform,  clean  (incl. anomaly  detection),  and load  into MySQL  \ndata  from  multiple  data  sources : road  traffic  sensors,  accidents , road  works,  weather .\nJun. 2014  – Sep.  2014  Data  Scientist   \n CONENTO  Madrid,  Spain  \n• Designed  an experiment  for Google  Spain  (conducted  in October  2014)  to measure  the impact  of \nYouTube  ads on the sales  of a car manufacturer 's dealer  network.  \n• A matched -pair,  cluster -randomized  design , which  involved  selecting  the test and control  groups  \nfrom  a sample  of 50+ cities  in Spain  (where  geo-targeted  ads were  possible)  based  on their  sales -\nwise  similarity  over  time,  using  wavelets  (and  R). \nMANAGEM ENT – SALES  (Electri cal Eng. ) \nFeb. 2009 – Aug. 2013 Head  of Sales,  Spain  & Portugal  – Test &Measurement  dept.\n YOKOGAWA  Madrid,  Spain  \n• Applied  analysis  of sales  and market  trends  to decide  the direction  of the department.  \n• Led a team  of 7 people .   \n2 of 2 Juan Jose Carin  \nData Scientist   \nMountain View , CA 94041  \n 650-336-4590  | juanjose.carin@gmail.com  \n linkedin.com/in/juanjosecarin  | juanjocarin.github.io  \n \n• Increased  revenue  by 6.3%,  gross  profit  by 4.2%,  and operating  income  by 146%,  and achieved  a 30%  \nratio  of new  customers  (3x growth),  by entering  new  markets  and improvi ng customer  service  and \ntraining .\nSALES (Electri cal Eng.  & Telecom. ) \nApr. 2008 – Jan. 2009 Sales  Engineer  – Test  & Meas urement  dept. \n YOKOGAWA  Madrid,  Spain  \n• Promoted  to head  of sales  after  5 months  leading  the sales  team.  \nSep. 2004 – Mar. 2008 Sales  & Application  Engineer  \n AYSCOM  Madrid,  Spain  \n• Exceeded  sales  target  every  year  from  2005  to 2007  (achieved  60%  of the target  in the first 3 months  \nof 2008 ). \nEDUCATION\nJul. 2002 – Jun. 2004 Tutor  of Differential  & Integral  Calculus,  Physics,  and Digital  Electronic  Circuits\n ACADEMIA  UNIVERSITARIA  Madrid,  Spain  \n• Highest -rated  professor  in student  surveys,  in 4 of the 6 terms.  \n• Increased  ratio of stud ents passing the course by 25%.  \nProjects   See juanjocarin.github.io  for additional  information\n2016  SmartCam  \nCapstone  Python , OpenCV , TensorFlow , AWS  (EC2, S3, DynamoDB ) \nA scalable  cloud -based  video  monitoring  system  that features  motion  detection,  face  counting,  and image  recognition.\n2015  Implementation  of the Shortest  Path  and PageRank  algorithms  with  the Wikipedia  graph  dataset  \nMachine  Learning  at Scale  Hadoop  MrJob,  Python , AWS  EC2, AWS  S3\nUsing  a graph  dataset  of almost  half a million  nodes.  \n2015  Forest  cover  type  prediction  \nMachine  Learning  Python , Scikit -Learn , Matplotlib  \nA Kaggle  competition : predictions  of the predominant  kind  of tree cover,  from  strictly  cartographic  variables  such  as elevation  \nand soil type,  using  random  forests,  SVMs,  kNNs,  Naive Bayes,  Gradient  Descent,  GMMs , …\n2015  Redefining  the job search  process  \nStoring  and Retrieving  Data  Hadoop  HDFS , Hive , Spark , Python , AWS  EC2, Tableau\nA pipeline  that combines  data  from  Indeed  API and the U.S. Census  Bureau  to select  the best  locations  for data  scientists  \nbased  on the number  of job postings,  housing  cost,  etc.\n2015  A fresh  perspective  on Citi Bike  \nData  Visualization  and Communication  Tableau , SQLite\nAn interactive  website  to visualize  NYC Citi Bike  bicycle sharing  service.\n2015  Investigating  the effect  of competition  on the ability  to solve  arithmetic  problems  \nField  Experiments  R \nA randomized  controlled  trial in which  300+  participants  were  assigned  to a control  group  or one of two test groups  to \nevaluate  the effect  of competition  (being  compared  to no one or someone  better  or worse).  \n2014  Prediction  of customer  churn  for a mobile  network  carrier  \nData  Mining  SAS\nPredictions  from  a sample  of 45,000+  customers,  using  tree decisions,  logistic  regression , and neural  networks.  \n2014  Different  models  of Harmonized  Index  of Consumer  Prices  (HICP)  in Spain  \nTime  Series  SPSS , Demetra+\nForecasts  based  on exponential  smoothing,  ARIMA,  and transfer  function  (using  petrol  price  as independent  variable)  models.  ",
#        'Jonathan Whitmore\nPhD, Senior Data Scientist, O’Reilly AuthorMountain View, CA\n+1 650-943-3715\nBJBWhit@gmail.com\nÍJonathanWhitmore.com\nJBWhit\nJonathanBWhitmore\nExperience\n2016–\nPresentSenior Data Scientist ,Silicon Valley Data Science , Mountain View, CA, USA.\n{Consulting as a member of several small data science/data engineering teams at multiple companies.\n{Creating output to explain data analysis, data visualization, and statistical modeling results to managers.\n{Developing Data Science best practices for team.\n{Modeling non-contractual churn on customer population.\n{Modeling survey data responses with ordinal logistic regression in R.\n{Analyzing and visualizing user behavior migration.\n2014–2016 Data Scientist ,Silicon Valley Data Science , Mountain View, CA, USA.\n2014Insight Data Science Postdoctoral Fellow ,Insight Data Science , Palo Alto, CA, USA.\n{Created a Data Science project to predict the auction sale price of Abstract Expressionist art.\n2011–2014 Postdoctoral Research Associate ,Swinburne University , Melbourne, AUS.\n{Cleaned noisy and inhomogeneous astronomical data taken over four years by diﬀerent observing groups.\n{Utilized numerous statistical techniques, including sensitivity analysis on non-linear propagation of\nerrors, Markov-Chain Monte Carlo for model building, and hypothesis testing via information criterion.\n{Simulated spectroscopic data to expose systematic errors that challenge long-standing results on whether\nthe fundamental physical constants of the universe are constant.\n2005–2011 Graduate Student Researcher ,UCSD, San Diego, CA, USA.\n{Developed a novel technique to extract information from high resolution spectroscopic data that led to\nuncovering unknown short-range systematic errors.\nProgramming and Development Skills\nLanguages Python, SQL (Impala/Hive), R, L ATEX, Bash.\nToolsJupyter Notebook, pandas, matplotlib, seaborn, numpy, scikit-learn, scipy, pymc3, git, pandoc.\nPublishing, Speaking, and Side Projects\n2017Instructor Stanford Continuing Studies: Tips and Tricks for Data Scientists: Optimizing Your Workﬂow.\n2017Invited Keynote: USC Career Conference Beyond the PhD.\n2016PyData SF: Mental Models to Use and Avoid as a Data Scientist.\n2016O’Reilly author: Jupyter Notebook for Data Science Teams [screencast], editor O’Reilly Media.\n2016UC Berkeley Master in Data Science Guest Lecturer: Jupyter Notebook Usage.\n2015OSCON Speaker: IPython Notebook best practices for data science.\n2013-2014 Contributor to astropy; creator of dipole_error, an astronomy Python module.\n2013Co-star and narrator of Hidden Universe, a 3D IMAX astronomy ﬁlm playing worldwide.\nEducation\n2011PhD Physics ,University of California San Diego , San Diego, CA, USA.\nThesis title: The Fine-Structure Constant and Wavelength Calibration.\n2005Bachelor of Science–Magna Cum Laude ,Vanderbilt University , Nashville, TN, USA.\nTriple major: Philosophy; Mathematics; Physics (honors).']
#keyword_path = [{'skill': 'Deep Learning', 'skills': ['PyTorch', 'Tensorflow']},{'skill': 'Machine Learning', 'skills': ['Python', 'R']}]

#new_data = compare_resumes_to_job(resumes_path, keyword_path)

#skills = ['Deep Learning', 'Machine Learning']
#new_data = sort_applicants(new_data, skills)
#create_plot_resume_to_job(new_data)