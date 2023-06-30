
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
import matplotlib.pyplot as plt

from create_vectorstore_resume import create_resume_vectorstore
from create_job_summary import create_vectorstore, make_chain


################### Create job vectorstore ###################
job_directory="C:/Users/SEPA/lanchain_ir2/Job_data/Research Scientist - NLP.txt"
create_vectorstore(job_directory)
chat_history = []
chain = make_chain()
question_job_1 = f"Please summarise the technical Corresponding skill catalogue and profile needed for this job? Please return the answer in a concise manner, no more than 250 words. If not found, return 'Not provided'"
# question_job_2 = f"What are the requirements needed for this job? Please return the answer in a concise manner, no more than 250 words. If not found, return 'Not provided'"
response = chain({"question": question_job_1, "chat_history": chat_history})
needed_Corresponding skill catalogue = response['answer']

##################### Create resume vectorstore ################
create_resume_vectorstore('C:/Users/SEPA/lanchain_ir2/Resume_data_pdf/')
embedding = OpenAIEmbeddings(openai_api_key="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")
resume_vector_store = Chroma(
    collection_name="resume-embeddings",
    embedding_function=embedding,
    persist_directory="embeddings/chroma",
)

retriever = resume_vector_store.as_retriever(search_kwargs={"k": 3})
# docs = retriever.get_relevant_documents("I have one year of experience with NLP and MLOps. Moreover I have worked with AWS, Kubernetes and Docker.")
docs = retriever.get_relevant_documents(needed_Corresponding skill catalogue)

docs_score = resume_vector_store.similarity_search_with_score(query=needed_Corresponding skill catalogue, distance_metric="cos", k = 3)

applicant_values = []
score_values = []
for doc in docs_score:
    applicant_values.append(doc[0].metadata["source"].split('\\')[-1].split('.')[0:-1])
    score_values.append(doc[1])

data = pd.DataFrame({'Applicant': applicant_values, 'Score': score_values})

fig, ax = plt.subplots(figsize=(10, 10))
data.plot.bar(x='Applicant', y='Score', ax=ax)
ax.set_xlabel('Applicants')
ax.set_ylabel('Cosine distance')
plt.tight_layout()
plt.show()




