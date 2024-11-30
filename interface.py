import streamlit as st
from crawlerThread import *
from facultyParser import *
from index_gen import *
from search_engine import *


class Wrapper():
    def __init__(self):
        self.vectorizer = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.model = None
        self.index = None
        self.keys = None

    def clear_database(self):
        # connect to local server
        DB_NAME = "CPP"
        DB_HOST = "localhost"
        DB_PORT = 27017
        try:
            client = MongoClient(host=DB_HOST, port=DB_PORT)
            db = client[DB_NAME]
            # delete all collections in database to start fresh
            db['pages'].drop()
            db['faculty'].drop()
            db['embeddings'].drop()
            db['inverted_index'].drop()
            print("Database cleared successfully")
        except:
            print("Database not connected successfully")

    def run_crawler(self):
        crawler = Crawler('https://www.cpp.edu/sci/biological-sciences/index.shtml')
        crawler.crawl()
        print("Crawling completed!")

    def run_parser(self):
        db = connectDataBase()
        pages_collection = db['pages']
        faculty_collection = db['faculty']
        process_faculty_pages(pages_collection, faculty_collection)
        print("Faculty Pages parsing completed!")

    def run_index_gen(self):
        create_and_store_index_and_embeddings()
        # llm_embedding = generate_embeddings_llma()
        llm_embedding = generate_embeddings()
        index, doc_ids = hnsw_index(llm_embedding)
        save_hnsw_index(index, doc_ids, "index.bin", "keys.pkl")
        print("Index generation completed!")

    def generate_interface(self):
        try:
            with open(VECTORIZER_FILE, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            return f"Vectorizer file {VECTORIZER_FILE} not found."
        
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        
        self.index, self.keys = load_hnsw_index(INDEX_PATH, KEYS_PATH)

        query_embedding("test", self.model, self.tokenizer, self.device , self.index, self.keys, 5)
        
        st.title("CPP Biology Department Search Engine")
        st.write("This is a search engine for the CPP Biology Department. You can use this to search for faculty members and their research interests.")
        with st.sidebar:
            st.image("images/cpp-logo.png",width=250)
            st.subheader("About", divider="gray")
            st.write(
                    "Our team has developed a search engine for Cal Poly Pomona's Biology department. Users can enter a query \
                    and discover which faculty members are most relevant to their search. To access the source code, please click\
                    the button below."
                )
            # Add link to GitHub repository
            st.link_button("GitHub Repo", "https://github.com/thvu02/SearchEngine")
        print("Interface generated!")

    @st.fragment
    def search(self):
        print("Searching...")
        query = st.text_input("Enter your query")

        if not query:
            st.write("Query cannot be empty. Please try again.")
        
        # Measure overall execution time
        overall_start_time = time.time()

        with ThreadPoolExecutor() as executor:
            # Run the functions in parallel
            future_tfidf = executor.submit(query_search_tfidf, query, self.vectorizer)
            future_llm = executor.submit(query_embedding, query, self.model, self.tokenizer, self.device, self.index, self.keys, 10)

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
            if res['similarity'] > 0.07:
                st.write(f"name = {res['name']}, similarity = {res['similarity']}, url = {res['url']}, \n{res['text_snippet']} \n")

if __name__ == "__main__":
    instance = Wrapper()
    instance.clear_database()
    instance.run_crawler()
    instance.run_parser()
    instance.run_index_gen()
    instance.generate_interface()
    instance.search() # should re-run this after every search