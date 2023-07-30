import re
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from dotenv import load_dotenv

import os
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

# In summary, the code replaces any occurrence of a newline character (\n) in the input text, except when it is
# immediately preceded and followed by another newline character, with a single space character. This can be useful for
# removing single line breaks while preserving paragraphs or distinct blocks of text in the content.
def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

# the code replaces any occurrence of two or more consecutive newline characters (\n\n, \n\n\n, etc.) in the input text
# with a single newline character (\n).
def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(pages, cleaning_functions):
    cleaned_pages = []
    for metadata, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((metadata, text))
    return cleaned_pages


def text_to_docs(text):
    """Converts list of strings to a list of Documents with metadata."""

    # text = cleaned_text_pdf
    doc_chunks = []

    for source, page in text:
        # page_num, source, page = text[0][0],text[0][1],text[0][2]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk": i,
                    "source": source,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks

def text_to_wholeDocs(text):
    """Converts list of strings to a list of Documents with metadata."""

    # text = cleaned_text_pdf
    doc_chunks = []

    for source, resume in text:

        doc = Document(
            page_content=resume,
            metadata={
                "source": source,
            },
        )
        doc_chunks.append(doc)

    return doc_chunks


def create_resume_vectorstore(resume_directory):
    load_dotenv()
    # resume_directory = 'C:/Users/SEPA/lanchain_ir2/Resume_data_pdf/'
    loader = DirectoryLoader(resume_directory, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    resumes=[]
    text = documents[0].page_content
    for i in range(1, len(documents)):
        if (documents[i].metadata['source'] == documents[i-1].metadata['source']):
            text += documents[i].page_content
        else:
            resumes.append((documents[i-1].metadata['source'], text))
            text = documents[i].page_content
        if (i == len(documents)-1):
            resumes.append((documents[i].metadata['source'], text))

    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text(resumes, cleaning_functions)
    # document_chunks = text_to_docs(cleaned_text_pdf)
    document_chunks = text_to_wholeDocs(cleaned_text_pdf)

    # Step 3 + 4: Generate embeddings and store them in DB
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name="resume-embeddings",
        persist_directory="embeddings/chroma",
    )
    vector_store.persist()