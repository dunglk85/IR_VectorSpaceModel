import sys

from load_data import LoadDataset 
from vsm import VectorSpaceModel

if __name__ == "__main__":
    corpus = sys.argv[1]

    if len(sys.argv > 2):
        queries = sys.argv[2]
        if len(sys.argv > 3):
            relevance = sys.argv[3]
            