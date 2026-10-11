from config import *
ignore_warnings()
import logging
import os
from services.rag.retriever import Retriever
from services.rag.generator import Generator
from services.rag.settings import RagSettings
from services.rag.client import DBClient
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false" # disabling chroma telemetry
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction, OllamaEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb import Collection


class QueryRag:
    """
    Handler class to performs RAG queries.

    Attributes:
        rag_db (RagDB): A RagDB instance.
        prompt (str): The template prompt for queries.
        model_choice (str, optional): The selected model.
    """

    def __init__(self,  settings: RagSettings, mode="production") -> None:
        """
        Constructor.

        Args:
            settings (RagSettings): The settings object.
            mode (str): The mode (test/production).
            rag_type (str): Choose between text, image or text_image RAG.

        Returns:
            None
        """

        # log activity
        rag_logger.info(f"Initialising {self.__class__.__name__}")

        # hyperparameters
        self.settings = settings
        if mode == "benchmarking":
            self.settings.pre_compute_captions = False # override for benchmarking only
        text_db, image_db, caption_db = self._setup_rag(mode)

        # initialise retriever and generator
        self.retriever = Retriever(self.settings, text_db, image_db, caption_db)
        self.generator = Generator(self.settings)


    def query(self, question: str) -> tuple[str, list[str], list[str]]:
        """
        Main method to query the vector DB.

        Attributes:
            question (str): The input question.

        Returns:
            tuple[str, list[str], list[str]]
        """

        answer: str
        sources: list[str]
        contexts: list[str]

        sources, contexts = self.retriever.retrieve_from_query(question)
        context = "\n\n------------\n\n".join(contexts)

        if self.settings.use_rewriter:
            rewriter = Generator(self.settings, question, rewriting=True)
            user_question = question
            question = rewriter.answer(user_question, context)

        answer = self.generator.answer(question, context)
        
        return answer, sources, contexts


    def _setup_rag(self, mode) -> tuple[Collection, Collection, Collection]:
        """
        Sets up the collections for queries.

        Attributes:
            question (str): The input question.

        Returns:
            tuple[str, list[str], list[str]]
        """

        client = DBClient(mode).client    
        ef_text = self.settings.text_embeddings_model
        
        text_db: Collection = client.get_collection(name="text", 
                        embedding_function=OllamaEmbeddingFunction(model_name=ef_text))
        image_db: Collection = client.get_collection(name="image",
                        embedding_function=OpenCLIPEmbeddingFunction(),
                        data_loader=ImageLoader())
        caption_db: Collection = client.get_collection(name="caption", 
                        embedding_function=OllamaEmbeddingFunction(model_name=ef_text))           
        
        return text_db, image_db, caption_db