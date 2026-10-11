import pytest

from config import test_logger
from services.rag.rag_db import RagDB
from services.rag.retriever import Retriever
from services.rag.settings import RagSettings
from services.rag.client import DBClient
from config import test_logger
from app import create_app
from app.extensions import db
from utility.db_setup.seed import Seed
from utility.rag_evaluation.rag_config import (TEXT, IMAGE, TEXT_IMAGE)

# logging settings - hiding unnecessary third-party INFO logs
logger = test_logger

rag_types = [TEXT, IMAGE, TEXT_IMAGE]

@pytest.fixture(scope="session")
def app_db():
    app = create_app(mode="test")

    # logging
    logger.info(f"Creating app...")

    with app.app_context():
        db.drop_all()
        db.create_all()
        seed = Seed(mode="test")
        seed.populate_db(db)

        yield app
        db.drop_all()


@pytest.fixture(scope="session")
def rag_db(app_db):
    rag_db: RagDB = RagDB(reset=True, mode="test")

    # logging
    logger.info(f"RagDB Folder: {rag_db.rag_db_folder}")

    img_db = rag_db.image_db
    text_db = rag_db.text_db
    caption_db = rag_db.caption_db
    client = DBClient(mode="test")

    yield rag_db, img_db, text_db, caption_db, client
    rag_db.clear_db()


@pytest.fixture(scope="session")
def settings():
    settings = RagSettings() # the default settings (production)
    settings.pre_compute_captions = True # making sure captions are created
    return settings


@pytest.mark.parametrize("rag_type", rag_types)
def test_retrieve_from_query_01(rag_db, rag_type, settings):
    """Checks returned sources and contexts for image RAG 
    overriding default settings by enabling reranking."""

    settings.rag_type = rag_type
    rag_db, img_db, text_db, caption_db, client = rag_db
    question = "Where is the JPMorgan skyscraper?"

    # default (no reranking)
    retriever = Retriever(settings, 
                          text_db=text_db, 
                          image_db=img_db, 
                          caption_db=caption_db)
    
    logger.info("Starting retrieval WITHOUT reranking...")
    sources, contexts = retriever.retrieve_from_query(question)

    # logging relevant values
    message = f"RETRIEVER FOR RAG TYPE: {rag_type}, NO RERANKING"
    _log_statements(message, question, sources, contexts)

    # with reranking
    settings_override = RagSettings()
    settings_override.use_reranker = True
    retriever = Retriever(settings_override, 
                          text_db=text_db, 
                          image_db=img_db, 
                          caption_db=caption_db)
    
    logger.info("Starting retrieval WITH reranking...")
    sources_r, contexts_r = retriever.retrieve_from_query(question)
    
    # logging relevant values
    message = f"RETRIEVER FOR RAG TYPE: {rag_type}, WITH RERANKING"
    _log_statements(message, question, sources_r, contexts_r)

    # assertion flags
    no_reranking_flag = _check_sources_contexts_exist(sources, contexts)
    reranking_flag = _check_sources_contexts_exist(sources_r, contexts_r)

    # assertion
    assert no_reranking_flag and reranking_flag


@pytest.mark.parametrize("rag_type", rag_types)
def test_retrieve_from_query_02(rag_db, rag_type, settings):
    """Querying retriever on default VS with captions."""

    settings.rag_type = rag_type
    rag_db, img_db, text_db, caption_db, client = rag_db
    question = "Where is the JPMorgan skyscraper?"

    # without CAPS
    retriever = Retriever(settings, 
                          text_db=text_db, 
                          image_db=img_db, 
                          caption_db=caption_db)
    
    logger.info("Starting retrieval WITHOUT captions...")
    sources, contexts = retriever.retrieve_from_query(question)

    # logging relevant values
    message = f"RETRIEVER FOR RAG TYPE: {rag_type} / WITHOUT CAPTIONS"
    _log_statements(message, question, sources, contexts)


    # with CAPS
    settings_override = RagSettings()
    settings_override.pre_compute_captions = True
    retriever_captions = Retriever(settings_override, 
                                    text_db=text_db, 
                                    image_db=img_db, 
                                    caption_db=caption_db)
    
    logger.info("Starting retrieval WITH captions...")
    sources_c, contexts_c = retriever_captions.retrieve_from_query(question)
    
    # logging relevant values
    message_c = f"RETRIEVER FOR RAG TYPE: {rag_type} / WITH CAPTIONS"
    _log_statements(message_c, question, sources_c, contexts_c)

    # assertion flags
    no_caps_flag = _check_sources_contexts_exist(sources, contexts)
    caps_flag = _check_sources_contexts_exist(sources_c, contexts_c)

    # assertion
    assert no_caps_flag and caps_flag



def _check_sources_contexts_exist(sources, contexts):
    # assertion flags
    has_sources = isinstance(sources, list) and len(sources) > 0
    has_contexts = isinstance(contexts, list) and len(contexts) > 0
    # return check
    return has_sources and has_contexts


def _log_statements(message, question, sources, contexts):
    # logging
    logger.info(f"MESSAGE: {message}")
    logger.info(f"QUESTION: {question}")
    logger.info(f"SOURCES: {sources}")
    logger.info(f"CONTEXTS: {contexts}")