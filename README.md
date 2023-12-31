# Introduction:
As an innovation-driven individual, I am excited about the ongoing AI revolution! In particular, the emergence of openAI, Langchain and Vectorstore solutions like Pineccone or ChromaDB has enabled new opportunities for building AI Applications. 
That's why I started this side project exploring the potential of Machine Learning in HR. Here I am working on the creation of competence profiles of applicants (based on CVs), job postings, and entire projects (based on Jira Stories) to enable the best possible matching.



# 1. Keyword-Search:

An exemplary insight into this project is provided by a module for competence screening or competence clustering of CVs, which is currently still in progress but which immediately demonstrates the potential of NLP techniques for recruiters. Currently I am integrating this feature in an application which consists of a React frontend and a Flask backend (which communicate via API requests), and enables recruiters, HR employees or Managers to analyze applicants regarding some predefines skillclusters and skills which are relevant for open job vaccancies. The recruiter can define skillclusters for which he want to search for and gets visualizations that gives a fast understanding of the skills and strength of an applicant. The user has also the option to sort the applicants according to specific skillclusters.

To start the Flask backend navigate to app.py (with any Terminal or Python IDE) and run the script app.py. Afterwards the React frontend can be launched by navigating to my-app and then running ‘npm start’. A page should now open in your browser where you can navigate to “Search for Skill Sets” and define skill clusters as tags, and upload some resumes.

![Selected skillclusters](defined_skillclusters.JPG)

Clicking the “Search applicant” button will create a chart showing how strongly the applicants' skills match the skill clusters (in percentages). 

![Alt Text](output_skillcluster.JPG)


The chart shows how strongly the applicants' skills (in percentages) match the skill clusters. As you can immediately see from the chart, the first applicant has a strong background in MLOps, while the second applicant has more experience with Data Engineering. This allows for quick screening of skill sets and their match with various projects and job openings.



# 2. Information Retrieval
For this feature I will utilize language embeddings to measure the cosine distance between a resume and a job description and therefore be able to rank the applicants. Unlike traditional keyword-based approaches that rely solely on matching keywords, language embeddings provide us with a more nuanced understanding of the text by capturing the contextual and semantic meaning.
Language embeddings are representations of words, phrases, or documents in a high-dimensional vector space, where words with similar meanings or contexts are located closer to each other. These embeddings are generated through advanced techniques such as word2vec, GloVe, or BERT, which leverage large amounts of textual data to learn the underlying patterns and relationships between words.

Following jupyter notebook shows a proof-of-concept of this feature: https://github.com/ascheppach/job_resume_tool/blob/main/Information%20Retrieval.ipynb

![image](https://github.com/ascheppach/job_resume_tool/assets/56842416/89e3beb2-41cf-4f21-b05a-4c63d5d01d07)

# 3. Information Extraction / Named Entity Recognition:
In this notebook I will show a proof-of-conecept how to train an Named Entity Recognition (NER) algorithm in order to be able to extract all relevant skills from an employee or applicant in a fully automated way. In the next step I want to use these automatically detected skills for the creation of employee and project competence profiles.

https://github.com/ascheppach/job_resume_tool/blob/main/Named%20Entity%20Recognition.ipynb


# 4. Topic Modeling
Utilizing topic modeling on a vast collection of textual data, including Jira stories, open jobs, resumes, and technical reports, provides valuable insights into emerging technical trends and identifies skill gaps and areas of expertise within an organization.
We will use the popular LDA algorithm, which is a probabilistic model that represents documents as a mixture of topics and which is a powerful technique used to uncover topics within a collection of documents. It assumes that each document is a combination of multiple topics, and each topic is a distribution of words. By analyzing a large collection of documents, LDA automatically discovers the underlying topics and their word distributions. This enables to uncover hidden thematic structures in text data and to detect specific skills and keywords belonging to a skill cluster.

https://github.com/ascheppach/job_resume_tool/blob/main/Topic%20Modeling.ipynb

![image](https://github.com/ascheppach/job_resume_tool/assets/56842416/cc43bc4c-808b-4736-81fd-d1afb856dafd)


# 5. Skill Clustering
Clustering algorithms such as k-means help to identify unique competency clusters and develop a competency tree. With this competence tree, specific competence areas can be explored in an intuitive way to understand trends and connections within the competence tree. Job descriptions, industry reports, resumes, and online job postings are particularly suitable as input data, as they provide insights into current market trends. The competency tree should be regularly updated and refined with new data to keep the competency tree current and reflect changes in industry trends.

https://github.com/ascheppach/job_resume_tool/blob/main/Skill%20Clustering.ipynb

![Alt Text](skill_tree.png)


# 6. Knowledge Graph
In our pursuit of understanding the intricate dynamics between applicants and job requirements, we delve into the construction of a knowledge graph. This powerful visual representation allows us to navigate the vast landscape of skills and competencies. At the core of our graph lie the main nodes representing applicants and job requirements, forming the foundation for a comprehensive understanding of their interplay. Skill nodes, intricately connected to both applicants and job requirements, provide us with valuable insights into the essential proficiencies sought in the industry. By unraveling the intricate web of connections within this knowledge graph, we aim to gain deeper insights into the alignment between applicants and job requirements, ultimately facilitating more informed decision-making processes.

https://github.com/ascheppach/job_resume_tool/blob/main/Knowledge%20Graph.ipynb

![Alt Text](knowledge_graph.png)

# 7. Future Work:
Integrate all provided features within my React-Flask Application to provide an interface that is easy to use (for manager, HR employees, recruiter etc.), provides queries and visualizations "per request", and enables quick insight into the relevant skill profiles of employees, projects, and applicants.







