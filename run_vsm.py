import sys

from load_data import LoadDataset 
from vsm import VectorSpaceModel

QUERY = " the crystalline lens in vertebrates including humans"
RELEVANT_DOCS = "12,13,14,71,78,137,141,163,164,165,166,167,168,169,170,171,179,180,181,182,183,184,185,210,211,498,499,500,501,502,503,505,506,507,509,510,512"
K = 10
PSEUDO = True

if __name__ == "__main__":
    corpus_file = sys.argv[1]
    
    # loading the dataset
    dataset = LoadDataset(corpus_file, "", "")

    docs = dataset.load_docs()

    vsm = VectorSpaceModel(docs)
    print("Number of documents in the collection: ", vsm.n_docs)
    print("Number of terms in the collection: ", vsm.n_terms)

    top10 = vsm.vector_space_model(QUERY, K)
    print(f"The relevance scores for the top {K} documents:")
    for d, s in list(top10.items()):
        print("DocID: ", d, "\tScore: ", s, "\n")

    if RELEVANT_DOCS != "":
        relevant = RELEVANT_DOCS.split(",")
        relevant = [int(r) for r in relevant]
        non_relevant = [i for i in range(vsm.n_docs) if i not in relevant]
        opt_query = vsm.relevance_feedback_rocchio(QUERY, relevant, non_relevant, alpha=1, beta=.75, gamma=.15)
        top10_rel = vsm.vector_space_model(opt_query, K)
        print(f"The relevance scores for the top {K} documents retrieved using relevance feedback:")
        for d, s in list(top10_rel.items()):
            print("DocID: ", d, "\tScore: ", s, "\n")
    
    if PSEUDO:
        opt_query_pseudo = vsm.pseudo_relevance_feedback(QUERY, K)
        top10_pseudo = vsm.vector_space_model(opt_query_pseudo, K)
        print(f"The relevance scores for the top {K} documents retrieved using pseudo relevance feedback:")
        for d, s in list(top10_pseudo.items()):
            print("DocID: ", d, "\tScore: ", s, "\n")
