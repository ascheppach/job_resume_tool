from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


# job_directory="C:/Users/SEPA/lanchain_ir2/Job_data/Research Scientist - NLP.txt"
api_key = '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'
def create_vectorstore(job_directory):
    # loader = DirectoryLoader(f'{root_dir}/Job_data/', glob="./*.txt", loader_cls=TextLoader)
    loader = TextLoader(job_directory)# , encoding='utf8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    job_chunks = text_splitter.split_text(documents[0].page_content)
    doc_chunks = []
    for i, chunk in enumerate(job_chunks):
        doc = Document(
            page_content=chunk
        )
        doc_chunks.append(doc)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    job_vector_store = Chroma.from_documents(
        doc_chunks,
        embeddings,
        collection_name="job_embeddings",
        persist_directory="job_embeddings/chroma",
    )

    job_vector_store.persist()


def make_chain():

    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0",
        openai_api_key=api_key
    )
    embedding = OpenAIEmbeddings(openai_api_key=api_key)

    job_vector_store = Chroma(
        collection_name="job_embeddings",
        embedding_function=embedding,
        persist_directory="job_embeddings/chroma",
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=job_vector_store.as_retriever(),
        return_source_documents=True,
        # verbose=True,
    )




