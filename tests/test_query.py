import pytest
from utility.rag_evaluation.rag_config import (BASELINE_K_SAMPLES, 
                                               TEXT, IMAGE, TEXT_IMAGE)
from config import test_logger
from services.rag.client import DBClient
from services.rag.rag_db import RagDB
from services.rag.query import QueryRag
from services.rag.settings import RagSettings
from app import create_app
from app.extensions import db
from utility.db_setup.seed import Seed

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
def settings_01():
    return RagSettings() # the default settings (production)


@pytest.fixture(scope="session")
def settings_02():
    settings = RagSettings() # the default settings (production)
    settings.pre_compute_captions = True
    return settings


@pytest.fixture(scope="session")
def settings_03():
    settings = RagSettings() # the default settings (production)
    settings.pre_compute_captions = True
    settings.use_reranker = True
    return settings


@pytest.fixture(scope="session")
def settings_04():
    settings = RagSettings() # the default settings (production)
    settings.pre_compute_captions = True
    settings.use_reranker = True
    settings.use_rewriter = True
    return settings


@pytest.mark.parametrize("rag_type", rag_types)
def test_query_rag_01(rag_db, rag_type, settings_01):
    """Tests query works"""

    settings_01.rag_type = rag_type
    rag = QueryRag(settings_01)
    question = "Where is the JPMorgan skyscraper?"
    answer, sources, contexts = rag.query(question)

    # logging
    logger.info(f"QUERYING RAG TYPE: {rag_type}")
    logger.info(f"Question:\n{question}")
    logger.info(f"Sources:\n{sources}")
    logger.info(f"Contexts:\n{contexts}")

    assert (answer is not None 
            and len(sources) == BASELINE_K_SAMPLES 
            and len(contexts) == BASELINE_K_SAMPLES)


@pytest.mark.parametrize("rag_type", rag_types)
def test_query_rag_02(rag_db, rag_type, settings_02):
    """Tests query works when captions are used."""
    settings_02.rag_type = rag_type
    rag = QueryRag(settings_02)
    question = "Where is the JPMorgan skyscraper?"
    answer, sources, contexts = rag.query(question)

    # logging
    logger.info(f"QUERYING RAG TYPE: {rag_type}")
    logger.info(f"Question:\n{question}")
    logger.info(f"Sources:\n{sources}")
    logger.info(f"Contexts:\n{contexts}")

    assert (answer is not None 
            and len(sources) == BASELINE_K_SAMPLES 
            and len(contexts) == BASELINE_K_SAMPLES)
    

@pytest.mark.parametrize("rag_type", rag_types)
def test_query_rag_03(rag_db, rag_type, settings_03):
    """Tests query works when captions are used and use_reranker == True"""
    settings_03.rag_type = rag_type
    rag = QueryRag(settings_03)
    question = "Where is the JPMorgan skyscraper?"
    answer, sources, contexts = rag.query(question)

    # logging
    logger.info(f"QUERYING RAG TYPE: {rag_type}")
    logger.info(f"Question:\n{question}")
    logger.info(f"Sources:\n{sources}")
    logger.info(f"Contexts:\n{contexts}")

    assert (answer is not None 
            and len(sources) == BASELINE_K_SAMPLES 
            and len(contexts) == BASELINE_K_SAMPLES)
    

@pytest.mark.parametrize("rag_type", rag_types)
def test_query_rag_04(rag_db, rag_type, settings_04):
    """Tests query works when captions, reranker and rewriter are used."""
    settings_04.rag_type = rag_type
    rag = QueryRag(settings_04)
    question = "Where is the JPMorgan skyscraper?"
    answer, sources, contexts = rag.query(question)

    # logging
    logger.info(f"QUERYING RAG TYPE: {rag_type}")
    logger.info(f"Question:\n{question}")
    logger.info(f"Sources:\n{sources}")
    logger.info(f"Contexts:\n{contexts}")

    assert (answer is not None 
            and len(sources) == BASELINE_K_SAMPLES 
            and len(contexts) == BASELINE_K_SAMPLES)