# Search Engine for CPP Biology Department
Our team as build a search engine specifically for Cal Poly Pomona's [Biology Department](https://www.cpp.edu/sci/biological-sciences/index.shtml). Users can enter arbitrary queries discover which faculty from the CPP Biology Department are most relevant to their query.

# Code Structure
```
- crawlerThread.py      (web crawler)
- facultyParser.py      (parse web crawler output)
- index_gen.py          (build inverted-index)
- search_engine.py      (search for relevant pages based on query)
- interace.py           (UI + wrapper for all scripts)
- images                (store images for UI)
- environemnt.yml      (list of dependecies for anaconda)
```

# To Run this Project
- Ensure you have all dependecies downloaded (check environemnt.yml)
- Run the following code in your terminal ```streamlit run interface.py```
- If you get an error involving HuggingFace
    - It is likely because you do not have a token to run their Llama 3.2 models
    - Visit https://huggingface.co/meta-llama/Llama-3.2-1B, create an account, and request access.
        - Turnaround is ~1 hour.
    - Through HuggingFace's website, go to ```Access Tokens``` (https://huggingface.co/settings/tokens)
    - Create a new token with ```write``` access. Do not create a ```fine-grained``` token as it will not have the necessary permissions for this use-case.
    - Copy the key provided after creating the token.
    - In your terminal, run ```huggingface cli login``` and paste the key.
    - Respond ```yes``` to the following questions and now you should no longer see errors. 
- A local web page will open on your browser
- Wait for the crawler, parser, inverted-index to do their work (~2 min)
- Enter any arbitrary query(s) and enjoy the output