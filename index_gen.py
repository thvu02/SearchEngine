import pickle
import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import faiss


# MongoDB connection details
client = MongoClient('mongodb://localhost:27017/')  
db = client['Project']  # Database name
faculty_collection = db['faculty']  # Source collection name
inverted_index_collection = db['inverted_index']  # Collection for inverted index
embeddings_collection = db['embeddings']  # Collection for document embeddings

# File path for saving vectorizer
VECTORIZER_FILE = "vectorizer.pkl"

def create_and_store_index_and_embeddings():
    
    documents = []
    doc_ids = []
    for doc in faculty_collection.find({}, {"text": 1, "_id": 1}):  
        documents.append(doc['text'])
        doc_ids.append(str(doc['_id']))  

    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  
    tfidf_matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()

    # Save the vectorizer to the local drive
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {VECTORIZER_FILE}.")

    # Construct the inverted index
    inverted_index = defaultdict(list)
    for term_idx, term in enumerate(terms):
        for doc_idx in range(tfidf_matrix.shape[0]):
            score = tfidf_matrix[doc_idx, term_idx]
            if score > 0:  
                inverted_index[term].append({"document_id": doc_ids[doc_idx], "tfidf_score": score})

    # Store inverted index in MongoDB
    inverted_index_collection.delete_many({})  
    for term, docs in inverted_index.items():
        inverted_index_collection.insert_one({"term": term, "documents": docs})

    print("Inverted index saved to MongoDB.")

    # Convert TF-IDF matrix to document vectors and store them
    document_vectors = tfidf_matrix.toarray()  # 
    embeddings_collection.delete_many({})  
    for doc_idx, doc_id in enumerate(doc_ids):
        embeddings_collection.insert_one({
            "document_id": doc_id,
            "tfidf": document_vectors[doc_idx].tolist()  
        })

    print("Document tfidf embeddings saved to MongoDB.")


def generate_embeddings_llma():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model = AutoModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    device = torch.device("mps")
    model = model.to(device)

    def split_into_chunks(text, tokenizer, chunk_size=1024):
        tokens = tokenizer.encode(text)
        return [tokenizer.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    
    embeds = {}
    documents = faculty_collection.find()

    for doc in documents:
        doc_id = str(doc['_id'])
        text = doc.get('text', '')
        
        if not text.strip():
            print(f"No text available for document ID {doc_id}. Skipping...")
            continue

        text_chunks = split_into_chunks(text, tokenizer)
        chunk_embeddings = []

        for chunk in text_chunks:
            inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, padding="max_length", truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]
                attention_mask = inputs.attention_mask
                chunk_embedding = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                chunk_embeddings.append(chunk_embedding[0].cpu().numpy())

        if not chunk_embeddings:
            print(f"No valid embeddings for document ID {doc_id}. Skipping...")
            continue

        final_embedding = np.mean(chunk_embeddings, axis=0)
        final_embedding /= np.linalg.norm(final_embedding)

        embeds[doc_id] = final_embedding

    return embeds

def generate_embeddings():
    """
    Processes documents, divides them into chunks, and computes embeddings for each chunk.

    Args:
        documents (list of dict): List of document dictionaries. Each dict must have '_id' and 'text' keys.
        model (AutoModel): Pre-trained model from HuggingFace Transformers for computing embeddings.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the pre-trained model.

    Returns:
        dict: A dictionary with document IDs as keys and their aggregated embeddings as values.
    """

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    device = torch.device("mps")
    model = model.to(device)

    def mean_pooling(model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    embeds = {}
    max_length = tokenizer.model_max_length  # Typically 512 for most models
    documents = faculty_collection.find()
    for doc in documents:
        doc_id = str(doc['_id'])
        text = doc.get('text', '')

        if not text.strip():
            print(f"No text available for document ID {doc_id}. Skipping...")
            continue

        # Tokenize and split the document into chunks within max_length
        encoded_inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            stride=max_length // 4,  # Use stride to create overlapping chunks
            return_overflowing_tokens=True,
            padding=True
        )

        chunk_embeddings = []

        # Process each chunk independently
        for i in range(len(encoded_inputs["input_ids"])):
            input_ids = encoded_inputs["input_ids"][i].unsqueeze(0).to(device)
            attention_mask = encoded_inputs["attention_mask"][i].unsqueeze(0).to(device)

            with torch.no_grad():
                model_output = model(input_ids, attention_mask=attention_mask)
            
            chunk_embedding = mean_pooling(model_output, attention_mask)
            chunk_embeddings.append(chunk_embedding)

        if chunk_embeddings:
            final_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)

            # Normalize the final embedding
            final_embedding = F.normalize(final_embedding, p=2, dim=1)

            embeds[doc_id] = final_embedding.squeeze().cpu().numpy()

        else:
            print(f"No valid chunks for document ID {doc_id}. Skipping...")

    return embeds



# IndexFlatIP
def exact_search_index(embeddings):
    d = len(next(iter(embeddings.values())))  
    index = faiss.IndexFlatIP(d)  
    embedding_matrix = np.array(list(embeddings.values()), dtype='float32')
    index.add(embedding_matrix)  
    return index, list(embeddings.keys())


# HNSW Index
def hnsw_index(embeddings, num_neighbors=10):
    d = len(next(iter(embeddings.values())))  
    index = faiss.IndexHNSWFlat(d, num_neighbors, faiss.METRIC_INNER_PRODUCT)
    embedding_matrix = np.array(list(embeddings.values()), dtype='float32')
    index.add(embedding_matrix)  
    return index, list(embeddings.keys())


def save_hnsw_index(index, keys, index_file_path, keys_file_path):
    folder_path = "vector_index/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    # Save 
    faiss.write_index(index, folder_path + index_file_path)
    # Save the keys 
    with open(folder_path + keys_file_path, 'wb') as f:
        pickle.dump(keys, f)

def main():

    create_and_store_index_and_embeddings()
    # llm_embedding = generate_embeddings_llma()
    llm_embedding = generate_embeddings()
    index, doc_ids = hnsw_index(llm_embedding)
    save_hnsw_index(index, doc_ids, "index.bin", "keys.pkl")

main()
