from config import *
ignore_warnings()
from app import db, create_app
from services.rag.client import DBClient
from app.database import ProjectList, ProjectReader, ProjectImageReader
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false" # disabling chroma telemetry
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
import chromadb
from chromadb import Collection
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import (OpenCLIPEmbeddingFunction, 
                                                OllamaEmbeddingFunction)
from services.rag.settings import RagSettings
from services.rag.caption_generator import CaptionGenerator


class RagDB:
    """
    Builds and updates the RAG vector database.

    Attributes:
        chunks (str)
        mode (str)
        settings (RagSettings)
        rag_db_folder (Path)
        txt_db_file (Path)
        client (DBCLient)
        cap_gen (CaptionGenerator)
        app (Flask)
    
    """

    def __init__(self, reset=False, mode="production", 
                 settings: RagSettings=RagSettings()) -> None:
        """
        Constructor.

        Args:
            reset (bool): Reset option to rebuild the database.
            mode (str): The mode (test/production).
            settings (RagSettings): The settings object.

        Returns:
            None
        """
        
        # log activity
        rag_logger.info(f"Initialising {self.__class__.__name__}")

        self.chunks = []
        self.mode = mode
        self.settings = settings
        self.client: chromadb.PersistentClient = DBClient(mode=mode).client
        self.cap_gen = CaptionGenerator(settings)
        
        if mode == "production":
            self.rag_db_folder = SERVICES_RAG_DB
            self.txt_db_file = SERVICES_RAG_DB_TEXT
        
        elif mode == "test":
            self.rag_db_folder = TEST_RAG_DB
            self.txt_db_file = TEST_RAG_DB_TEXT

        elif mode == "benchmarking":
            self.rag_db_folder = RAG_BENCH_EVAL_DB
            self.txt_db_file = RAG_BENCH_DB_TEXT
            self.settings.pre_compute_captions = False # override for benchmarking only

        if reset:
            # creating the app to provide context for app db fetching
            self.reset_rag_db()


    def reset_rag_db(self) -> None:
            
            # benchmarking mode does not need flask app db access
            if self.mode != "benchmarking":
                self.app = create_app(self.mode)

            self.clear_db() # wiping out entire db to rebuilding
            self.text_db, self.image_db, self.caption_db = self._create_db()
            self.seed_rag_db()


    def clear_db(self) -> None:
        """
        Clears the entire rag db.

        Returns:
            None
        """
        # log activity
        rag_logger.info(f"Clearing entire RAG Vector DB on mode {self.mode}.")
        self.client.reset()


    def seed_rag_db(self) -> None:
        """
        Populates the database.

        Returns:
            None
        """

        # text loading
        self._add_text_to_rag_db()
        self._export_chunks_to_file(self.chunks) # added line to get a .txt file with all chunks
        
        # image and caption loading (only applicable to architectural dataset)
        if self.mode != "benchmarking":
            self._add_images_to_rag_db()


    def _create_db(self) -> tuple[Collection, Collection]:
        """
        Initialises the text and image collections.

        Returns:
            tuple[Collection, Collection]
        """

        # embedding functions
        ef_text = self._get_embedding("text")
        ef_image = self._get_embedding("image")

        text_db: Collection = self.client.create_collection(name="text", 
                                                            embedding_function=ef_text)

        image_db: Collection = self.client.create_collection(name="image",
                                                             embedding_function=ef_image,
                                                             data_loader=ImageLoader())

        caption_db: Collection = self.client.create_collection(name="caption", 
                                                               embedding_function=ef_text)

        return text_db, image_db, caption_db


    def _get_embedding(self, embedding_type) -> OllamaEmbeddings | OpenCLIPEmbeddingFunction:
        """
        Creates the embedding function.

        Returns:
            OllamaEmbeddings or OpenCLIPEmbeddingFunction.
        """

        if embedding_type == "text":
            embeddings = OllamaEmbeddingFunction(
                model_name=self.settings.text_embeddings_model)

        elif embedding_type == "image":
            embeddings = OpenCLIPEmbeddingFunction()
        
        return embeddings


    def _add_text_to_rag_db(self) -> None:
        """
        Adds new chunks to the rag db.

        Returns:
            None
        """

        # benchmarking mode to compare with a public dataset of choice
        if self.mode == "benchmarking":

            from utility.rag_evaluation.benchmarker import Benchmarker
            bm = Benchmarker()
            for (id, document) in zip(bm.ids, bm.documents):
                self.text_db.add(ids=id, documents=document)
            return

        text_docs = self._create_text_docs_from_project()
        current_items = self.text_db.get(include=[])
        current_ids = set(current_items["ids"])

        # log activity
        rag_logger.info(f"Number of current documents in RAG (Text) DB: {len(current_ids)}")

        # adding documents not yet on the db
        self.chunks = [chunk for chunk in text_docs if chunk.metadata["id"] not in current_ids]

        if len(self.chunks):
            # log activity
            rag_logger.info(f"Added new documents: {len(self.chunks)}")

            added_chunk_ids = [chunk.metadata["id"] for chunk in self.chunks]
            self.text_db.add(ids=added_chunk_ids,
                             documents=[doc.page_content for doc in self.chunks], 
                             metadatas=[doc.metadata for doc in self.chunks])
        else:
            # log activity
            rag_logger.info("No documents to add.")


    def _add_images_to_rag_db(self) -> None:
        """
        Adds new chunks to the rag db.

        Returns:
            None
        """

        # benchmarking RAGBench dataset does not contain images
        if self.mode == "benchmarking":
            return

        imgs = self._create_img_docs_from_project()
        ids = [img.metadata["id"] for img in imgs]
        uris = [img.metadata["uri"] for img in imgs]

        # all I need is the image ids and their uris
        self.image_db.add(ids=ids, uris=uris)
        # adding all captions to caption_db collection
        self.cap_gen.add_captions_to_rag_db(self.image_db, self.caption_db)


    def _create_text_docs_from_project(self) -> list[Document]:
        """
        Creates documents from project summaries read from app db.

        Returns:
            list[Document]
        """

        with self.app.app_context():
            project_list = ProjectList(db.session)
            chunks : list[list[Document]] = []

            for project_id in project_list.ids:
                project_text_docs : list[Document] = []
                p = ProjectReader(db.session, project_id)

                project_text_docs.append(Document(page_content=p.summary,
                            metadata={"project_id": p.id}))
                partial_chunks = self._split_docs(project_text_docs)
                for idx, c in enumerate(partial_chunks):
                    c_id = str(c.metadata["project_id"]) + "_" + str(idx)
                    c.metadata["id"] = c_id
                chunks.append(partial_chunks)

            chunks_flat = [doc for sublist in chunks for doc in sublist]

            return chunks_flat


    def _create_img_docs_from_project(self) -> list[Document]:
        """
        Creates image docs from app db projects.

        Returns:
            list[Document]

        """

        with self.app.app_context():
            project_list = ProjectList(db.session)
            imgs : list[Document] = []

            for project_id in project_list.ids:
                project_images = ProjectImageReader(db.session, project_id)
                img_names = project_images.image_names
                img_paths = project_images.image_paths

                for img_name, img_path in zip(img_names, img_paths):
                    
                    img = Document(page_content="",
                                   metadata={"project_id": project_id,
                                             "id" : img_name,
                                             "uri" : "app/static/" + img_path,
                                             "type" : "image"})
                    imgs.append(img)

            return imgs


    @staticmethod
    def _split_docs(documents: list[Document]) -> list[Document]:
        """
        Splits the documents.

        Args:
            documents (list[Document]): The input documents.
        
        Returns:
            list[Document]
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents(documents)


    # exports the chunks to a .txt file
    def _export_chunks_to_file(self, chunks: list[Document]) -> None:
        """
        Exports all chunks to a .txt file (useful to build up questions 
        and references for evaluation tasks).

        Args:
            chunks (list[Document]): The text chunks.
        
        Returns:
            list[Document]
        """
                
        with open((self.txt_db_file), "w", encoding="utf-8", newline="\n") as f:
            for chunk in chunks: 
                f.write("\n----------\n")
                f.write(chunk.metadata["id"])
                f.write("\n\n")
                f.write(chunk.page_content)
