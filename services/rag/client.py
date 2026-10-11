from config import *
ignore_warnings()
from chromadb import PersistentClient, Settings

class DBClient:
    """
    Wrapper to Chroma DB client with mode configuration.

    Attributes:
        client (PersistentClient): The client for access to the vector DB.
    """

    def __init__(self, mode="production") -> None:
        """
        Constructor.

        Args:
            mode (str): The mode selector (test/production).

        Returns:
            None
        """
                
        # log activity
        rag_logger.info(f"Initialising {self.__class__.__name__}")
        
        if mode == "test":
            rag_db_folder = TEST_RAG_DB

        elif mode == "benchmarking":
            rag_db_folder = RAG_BENCH_EVAL_DB

        else:
            rag_db_folder = SERVICES_RAG_DB

        self.client: PersistentClient = PersistentClient(path=rag_db_folder, 
                                        settings=Settings(allow_reset=True))