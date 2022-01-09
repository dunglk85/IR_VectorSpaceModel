# Information Retrieval - Vector Space Model

Repository for the Information Retrieval exam final project.

Repository organization:

- `med_data` contains the [dataset](http://ir.dcs.gla.ac.uk/resources/test_collections/medl/) used, a collection of 1032 articles from a medical journal.
    - `MED.ALL` contains the documents collection,
    - `MED.QRY` contains a list of 30 queries,
    - `MED.REL` contains for each query in `MED.QRY` the list of known relevant documents, in the format `query_id   0   doc_id   1`

- `load_data.py`: script with the class `LoadDataset()` used to load the corpus of documents, and the queries and relevance documents files, if any.

- `vsm.py`: script with the class `VectorSpaceModel()` used to perform the retrieval. It contains functions to compute the TF-IDF for each term in each document, to vectorize documents and queries and to perform relevance and pseudo-relevance feedback. It contains also a function to perform standard preprocessing of terms and a function to evaluate the retrieval, given a set of queries and known associated relevant documents.

- `ranked_retrieval.ipynb`: notebook with all the implemented functions shown at work, on the Medline dataset. It shows also an evaluation in the performance of the program, through the computation of precision, recall and mean average precision.

- `run_vsm.py`: script to run the program by command line, giving as argument the corpus of documents: `python run_vsm.py med_data/MED.ALL`. The user can modify the script by inserting:
    - the required query, as the `QUERY` variable,
    - a value of `K`, corresponding to how many documents will be returned by the program,
    - a set of known relevant documents as the `RELEVANT_DOCS` variable, to allow the program to perform also relevance feedback,
    - `PSEUDO=True` if the user wishes to perform also pseudo-relevance feedback.
