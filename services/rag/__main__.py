from config import *
ignore_warnings()
import argparse
from services.rag.query import QueryRag
from services.rag.settings import RagSettings
# from utility.rag_evaluation.profiler import profiler


def main(query, mode="production", reset=False) -> None:
    """
    Executes the RAG service under selected configurations.

    """

    settings= RagSettings()
    
    if reset:
        from services.rag.rag_db import RagDB
        RagDB(reset, mode=mode) # implement hyperparameters also for RagDB
    
    query_rag = QueryRag(settings, mode=mode)
    query_rag.query(query)


if __name__ == "__main__":

    # arguments for running different configurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", 
                        type=lambda x: True if x.lower() in ("true") else False,
                        default="False", 
                        help="Reset the Chroma DB.")
    parser.add_argument("--mode", default="production", help="Choose production or testing.")
    parser.add_argument("--type", default="text", help="RAG type selection (text/image/text_image).")
    args = parser.parse_args()

    # run RAG, showing a sample query on the terminal
    sample_query = "A large glass building with a green lawn in front of it" # a literal match to a caption
    main(sample_query, mode=args.mode, reset=args.reset)