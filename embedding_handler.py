import pickle

def store_embeddings(docs, embeddings, sotre_name, path):
    vectorStore = FAISS.from_documents(docs, embeddings)

    with open(f"{path}/faiss_{sotre_name}.pkl", "wb") as f:
        pickle.dump(vectorStore, f)

def load_embeddings(sotre_name, path):
    with open(f"{path}/faiss_{sotre_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore