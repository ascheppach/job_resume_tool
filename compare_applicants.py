import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.chains.question_answering import load_qa_chain


def ask_openAI(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].text.strip()



applicant = 'CV_Scheppach.pdf'
job = 'Research Scientist - NLP.txt'
def get_skill_match(applicant, job):

    loader = PyPDFLoader(f"C:/Users/SEPA/lanchain_ir2/Resume_data_pdf/{applicant}")
    resume = loader.load()

    loader_job = TextLoader(f"C:/Users/SEPA/lanchain_ir2/Job_data/{job}")
    job = loader_job.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )

    for idx in range(len(job)):
        # idx=0
        if idx > 0:
            job_text += job[idx].page_content
        else:
            job_text = job[idx].page_content

    for idx in range(len(resume)):
        # idx=0
        if idx > 0:
            resume_chunks += text_splitter.split_text(resume[idx].page_content)
        else:
            resume_chunks = text_splitter.split_text(resume[idx].page_content)

    embeddings = OpenAIEmbeddings(openai_api_key="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")
    knowledge_base = FAISS.from_texts(resume_chunks, embeddings)
    question = f"What are the candidate's technical skills? Please return the answer in a concise manner, no more than 350 words. If not found, return 'Not provided'"
    docs = knowledge_base.similarity_search(question)
    llm = OpenAI(openai_api_key="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", temperature=0.0, model_name="text-davinci-003", max_tokens="2000")
    chain = load_qa_chain(llm, chain_type="stuff")
    resume_summary = chain.run(input_documents=docs, question=question)

    summary_question = f"Job requirements: {{{job_text}}}" + f"Applicant skills: {{{resume_summary}}}" + "Please return a summary about how well the skills of the applicant and the requirments of this job match together (limited to 300 words);'"
    matching_summary = ask_openAI(summary_question)

    return matching_summary



applicant = 'CV_Scheppach.pdf'
def get_workExperience_and_Degree(applicant):

    loader = PyPDFLoader(f"C:/Users/SEPA/lanchain_ir2/Resume_data_pdf/{applicant}")
    resume = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )

    for idx in range(len(resume)):
        # idx=0
        if idx > 0:
            resume_chunks += text_splitter.split_text(resume[idx].page_content)
        else:
            resume_chunks = text_splitter.split_text(resume[idx].page_content)

    embeddings = OpenAIEmbeddings(openai_api_key="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")
    knowledge_base = FAISS.from_texts(resume_chunks, embeddings)
    llm = OpenAI(openai_api_key="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", temperature=0.0,
                 model_name="text-davinci-003", max_tokens="2000")
    chain = load_qa_chain(llm, chain_type="stuff")

    # get highest degree
    question = f"What is the candidate's highest degree? Please return the answer in a concise manner, no more than 100 words. If not found, return 'Not provided'"
    docs = knowledge_base.similarity_search(question)
    highest_degree = chain.run(input_documents=docs, question=question)

    # get years of experience
    question = f"What is the candidate's years of work experience? Please return the answer in a concise manner, no more than 150 words. If not found, return 'Not provided'"
    docs = knowledge_base.similarity_search(question)
    yearsWorkExperience = chain.run(input_documents=docs, question=question)

    return highest_degree, yearsWorkExperience


applicant = 'CV_Scheppach.pdf'
def get_applicantName(applicant):

    loader = PyPDFLoader(f"C:/Users/SEPA/lanchain_ir2/Resume_data_pdf/{applicant}")
    resume = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )

    for idx in range(len(resume)):
        # idx=0
        if idx > 0:
            resume_chunks += text_splitter.split_text(resume[idx].page_content)
        else:
            resume_chunks = text_splitter.split_text(resume[idx].page_content)

    embeddings = OpenAIEmbeddings(openai_api_key="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")
    knowledge_base = FAISS.from_texts(resume_chunks, embeddings)
    llm = OpenAI(openai_api_key="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", temperature=0.0,
                 model_name="text-davinci-003", max_tokens="2000")
    chain = load_qa_chain(llm, chain_type="stuff")

    # get highest degree
    question = f"What is the candidate's name? Please return the answer in a concise manner, no more than 50 words. If not found, return 'Not provided'"
    docs = knowledge_base.similarity_search(question)
    nameCandidate = chain.run(input_documents=docs, question=question)

    return nameCandidate


job_directory="C:/Users/SEPA/lanchain_ir2"
def create_and_store_job_summaries(job_directory):
    root_dir = job_directory
    # loader = DirectoryLoader(f'{root_dir}/Resume_data/', glob="./*.txt", loader_cls=TextLoader)
    loader = DirectoryLoader(f'{root_dir}/Job_data/', glob="./*.txt", loader_cls=TextLoader)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=30,
        length_function=len
    )

    # iterate over the documents, summarize them and store the summarised version
    for idx in range(len(documents)):
        # idx=0
        job = documents[idx].page_content
        job_chunks = text_splitter.split_text(job)

        embeddings = OpenAIEmbeddings(openai_api_key="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")
        knowledge_base = FAISS.from_texts(job_chunks, embeddings)

        question_job = f"What are the skills and profile needed for this job? Please return the answer in a concise manner, no more than 250 words. If not found, return 'Not provided'"

        # hier macht er eine Zusammenfassung eben: von den ursprünglich 20 job_chunks haben wir jetzt die 4 ähnlichsten übrig
        docs_job = knowledge_base.similarity_search(question_job)
        for i in range(len(docs_job)):
            # i=2
            if i > 0:
                string += docs_job[i].page_content
            else:
                string = docs_job[i].page_content

        file_name = documents[idx].metadata['source'].split("\\")[-1]
        file = open(file_name, 'w')
        file.write(string)
        file.close()


resume_directory = "C:/Users/SEPA/lanchain_ir2"
def create_and_store_resume_summaries(resume_directory):
    root_dir = resume_directory
    loader = DirectoryLoader(f'{root_dir}/Resume_data/', glob="./*.txt", loader_cls=TextLoader)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=30,
        length_function=len
    )

    # iterate over the documents, summarize them and store the summarised version
    for idx in range(len(documents)):
        # idx=0
        resume = documents[idx].page_content
        resume_chunks = text_splitter.split_text(resume)

        embeddings = OpenAIEmbeddings(openai_api_key="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")
        knowledge_base = FAISS.from_texts(resume_chunks, embeddings)

        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})
        # knowledge_base = FAISS.from_texts(chunks, embeddings)

        question_resume = f"What is this candidate's technical skills? Please return the answer in a concise manner, no more than 250 words. If not found, return 'Not provided'"

        # hier macht er eine Zusammenfassung eben: von den ursprünglich 20 job_chunks haben wir jetzt die 4 ähnlichsten übrig
        docs_resume = knowledge_base.similarity_search(question_resume)
        for i in range(len(docs_resume)):
            # i=2
            if i > 0:
                string += docs_resume[i].page_content
            else:
                string = docs_resume[i].page_content

        file_name = documents[idx].metadata['source'].split("\\")[-1]
        file = open(file_name, 'w')
        file.write(string)
        file.close()
