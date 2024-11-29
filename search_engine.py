import pymongo
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from bson.objectid import ObjectId  
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import faiss



client = pymongo.MongoClient()
db = client['Project']
faculty_collection = db['faculty']
inverted_index_collection = db['inverted_index']  
embeddings_collection = db['embeddings']

VECTORIZER_FILE = "vectorizer.pkl"
INDEX_PATH="vector_index/index.bin"
KEYS_PATH="vector_index/keys.pkl" 

llm_weight=0.4
tfidf_weight=0.6


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress logging from Transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

def cross_reference_results(combined_results, collection):

    enriched_results = []
    for result in combined_results:
        doc_id = result['document_id']
        similarity = result['similarity']

        try:
            # Convert string document ID back to ObjectId
            object_id = ObjectId(doc_id)
        except Exception as e:
            enriched_results.append({
                "document_id": doc_id,
                "similarity": similarity,
                "message": f"Invalid document ID format: {doc_id}. Error: {str(e)}"
            })
            continue

        # Fetch the document from the original collection
        document = collection.find_one({"_id": object_id})
        if not document:
            enriched_results.append({
                "document_id": doc_id,
                "similarity": similarity,
                "message": f"No details found for document ID: {doc_id}"
            })
            continue

        # Extract details and enrich the result
        professor_name = document.get("name", "Name not available")
        professor_url = document.get("url", "URL not available")
        enriched_results.append({
            "document_id": doc_id,
            "similarity": similarity,
            "name": professor_name,
            "url": professor_url
        })

    return enriched_results


def load_hnsw_index(index_file_path, keys_file_path):
    
    index = faiss.read_index(index_file_path)
    with open(keys_file_path, 'rb') as f:
        keys = pickle.load(f)
    return index, keys


def query_index(index, query_embedding, k=10, doc_ids=None):

    query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    if doc_ids is not None:
        result_ids = [doc_ids[idx] for idx in indices[0]]
        return result_ids, distances[0]
    return indices[0], distances[0]


def query_embedding(query, model, tokenizer, device ,index, keys, k):

    # def split_into_chunks(text, tokenizer, chunk_size=1024):

    #     tokens = tokenizer.encode(text)
    #     return [tokenizer.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

    # text_chunks = split_into_chunks(query, tokenizer)
    # chunk_embeddings = []

    # for chunk in text_chunks:
    #     inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, padding="max_length", truncation=True).to(device)
    #     with torch.no_grad():
    #         outputs = model(**inputs, output_hidden_states=True)
    #         hidden_states = outputs.hidden_states
    #         last_hidden_state = hidden_states[-1]
    #         attention_mask = inputs.attention_mask


    #         chunk_embedding = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    #         chunk_embeddings.append(chunk_embedding[0].cpu().numpy())

    # if not chunk_embeddings:
    #     print("No valid embeddings for the query.")
    #     return None

    # query_embedding = np.mean(chunk_embeddings, axis=0)
    # query_embedding /= np.linalg.norm(query_embedding)  


    # results = query_index(index, query_embedding, k, keys)

    # return results
    def mean_pooling(model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Split the query into manageable chunks
    encoded_inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
        stride=tokenizer.model_max_length // 4,  # Overlap chunks for better context
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

    if not chunk_embeddings:
        print("No valid embeddings for the query.")
        return None

    # Aggregate chunk embeddings into a single query embedding
    query_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)

    # Normalize the query embedding
    query_embedding = F.normalize(query_embedding, p=2, dim=1).squeeze(0).cpu().numpy()

    # Perform similarity search using the query embedding
    results = query_index(index, query_embedding, k, keys)

    return results


def query_search_tfidf(query_sentence, vectorizer):
    query_vector = vectorizer.transform([query_sentence]).toarray()
    query_terms = vectorizer.inverse_transform(query_vector)[0]  
    candidate_docs = defaultdict(float)
    document_embeddings = {}

    for term in query_terms:  
        term_entry = inverted_index_collection.find_one({"term": term})
        if term_entry:
            for doc in term_entry['documents']:
                doc_id = doc['document_id']
                candidate_docs[doc_id] += doc['tfidf_score']
    

    for doc_id in candidate_docs.keys():
        embedding_entry = embeddings_collection.find_one({"document_id": doc_id})
        if embedding_entry:
            document_embeddings[doc_id] = np.array(embedding_entry["tfidf"])
    

    results = []
    for doc_id, embedding in document_embeddings.items():
        similarity = cosine_similarity(query_vector, embedding.reshape(1, -1))[0][0]
        results.append({"document_id": doc_id, "similarity": similarity})

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    return results



# def combine_rankings(ranking1, ranking2, weight1=0.5, weight2=0.5):
#     ranking1_dict = {item['document_id']: item['similarity'] for item in ranking1}
#     if ranking1_dict:
#         min_val, max_val = min(ranking1_dict.values()), max(ranking1_dict.values())
#         if max_val > min_val:  
#             normalized_ranking1 = {doc_id: (sim - min_val) / (max_val - min_val) * 0.99 + 0.01
#                                    for doc_id, sim in ranking1_dict.items()}
#         else:
#             normalized_ranking1 = {doc_id: 0.5 for doc_id in ranking1_dict} 
#     else:
#         normalized_ranking1 = {}

#     ranking2_doc_ids, ranking2_similarities = ranking2
#     if ranking2_doc_ids:
#         min_val, max_val = ranking2_similarities.min(), ranking2_similarities.max()
#         if max_val > min_val:  # Avoid division by zero
#             normalized_ranking2 = {doc_id: (sim - min_val) / (max_val - min_val) * 0.99 + 0.01
#                                    for doc_id, sim in zip(ranking2_doc_ids, ranking2_similarities)}
#         else:
#             normalized_ranking2 = {doc_id: 0.5 for doc_id in ranking2_doc_ids}  
#     else:
#         normalized_ranking2 = {}

#     combined_scores = {}
#     all_doc_ids = set(normalized_ranking1.keys()).union(normalized_ranking2.keys())
#     for doc_id in all_doc_ids:
#         score1 = normalized_ranking1.get(doc_id, 0.0)  
#         score2 = normalized_ranking2.get(doc_id, 0.0)  
#         combined_scores[doc_id] = weight1 * score1 + weight2 * score2

#     sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
#     return [{'document_id': doc_id, 'similarity': score} for doc_id, score in sorted_combined]



# def combine_rankings(ranking1, ranking2, weight1=0.5, weight2=0.5):
#     import numpy as np
    
#     # Extract scores from ranking1
#     ranking1_dict = {item['document_id']: item['similarity'] for item in ranking1}
#     if ranking1_dict:
#         ranking1_scores = np.array(list(ranking1_dict.values()))
#         mean1, std1 = ranking1_scores.mean(), ranking1_scores.std()
#         z_scores_ranking1 = {doc_id: (score - mean1) / (std1 if std1 > 0 else 1)
#                              for doc_id, score in ranking1_dict.items()}
#     else:
#         z_scores_ranking1 = {}

#     # Extract scores from ranking2
#     ranking2_doc_ids, ranking2_similarities = ranking2
#     if ranking2_doc_ids:
#         mean2, std2 = ranking2_similarities.mean(), ranking2_similarities.std()
#         z_scores_ranking2 = {doc_id: (score - mean2) / (std2 if std2 > 0 else 1)
#                              for doc_id, score in zip(ranking2_doc_ids, ranking2_similarities)}
#     else:
#         z_scores_ranking2 = {}

#     # Combine rankings with weighted average
#     combined_scores = {}
#     all_doc_ids = set(z_scores_ranking1.keys()).union(z_scores_ranking2.keys())
#     for doc_id in all_doc_ids:
#         score1 = z_scores_ranking1.get(doc_id, 0.0)  # Missing documents get 0
#         score2 = z_scores_ranking2.get(doc_id, 0.0)  # Missing documents get 0
#         combined_scores[doc_id] = weight1 * score1 + weight2 * score2

#     # Sort by combined similarity scores in descending order
#     sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

#     return [{'document_id': doc_id, 'similarity': score} for doc_id, score in sorted_combined]


def combine_rankings(ranking1, ranking2, weight1=0.5, weight2=0.5):
    ranking1_dict = {item['document_id']: item['similarity'] for item in ranking1}
    ranking2_doc_ids, ranking2_similarities = ranking2
    ranking2_dict = {doc_id: score for doc_id, score in zip(ranking2_doc_ids, ranking2_similarities)}

    max_score1 = max(ranking1_dict.values(), default=1)
    max_score2 = max(ranking2_dict.values(), default=1)

    combined_scores = {}
    all_doc_ids = set(ranking1_dict.keys()).union(ranking2_dict.keys())
    for doc_id in all_doc_ids:
        score1 = ranking1_dict.get(doc_id, 0.0)
        score2 = ranking2_dict.get(doc_id, 0.0)
        combined_scores[doc_id] = (weight1 / max_score1) * score1 + (weight2 / max_score2) * score2

    # Sort by combined similarity scores in descending order
    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    return [{'document_id': doc_id, 'similarity': score} for doc_id, score in sorted_combined]


def main_interface():

    try:
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        return f"Vectorizer file {VECTORIZER_FILE} not found."
    


    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    # model = AutoModel.from_pretrained(model_name)
    # model.resize_token_embeddings(len(tokenizer))
    # device = torch.device("mps")
    # model = model.to(device)

    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    device = torch.device("mps")
    model = model.to(device)
    

    index, keys = load_hnsw_index(INDEX_PATH, KEYS_PATH)

    query_embedding("test", model, tokenizer, device ,index, keys, 5)

    print("a. Enter Query \nb. Exit")

    while True:
        # Display options
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'b':
            print("Thank you for using the Search System. Goodbye!")
            break

        elif choice == 'a':
            query = input("\nEnter your query: ").strip()
            if not query:
                print("Query cannot be empty. Please try again.")
                continue
            
            print(f"\nPerforming query search for: '{query}'\n")


            # Measure overall execution time
            overall_start_time = time.time()

            with ThreadPoolExecutor() as executor:
                # Run the functions in parallel
                future_tfidf = executor.submit(query_search_tfidf, query, vectorizer)
                future_llm = executor.submit(query_embedding, query, model, tokenizer, device, index, keys, 10)

                # Wait for results
                tfidf_results = future_tfidf.result()
                llm_results = future_llm.result()

            overall_end_time = time.time()
            # print(overall_end_time - overall_start_time)

            # print(tfidf_results)
            # print()
            # print(llm_results)

            
            results = combine_rankings(tfidf_results, llm_results, tfidf_weight, llm_weight)

            final_res = cross_reference_results(results, faculty_collection)

            # # print(final_res)
            # # print("\n\n")

            for res in final_res:
                if res['similarity'] > 0.35:
                    print(f"name = {res['name']}, similarity = {res['similarity']}, url = {res['url']}")

        else:
            print("Invalid choice.")

# Start the user interface
main_interface()

