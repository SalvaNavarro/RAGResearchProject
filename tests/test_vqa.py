import pytest
from services.rag.rag_db import RagDB
from services.rag.vqa import VQA
from utility.rag_evaluation.rag_config import (BASELINE_K_SAMPLES)
from config import test_logger
from services.rag.settings import RagSettings
from services.rag.client import DBClient
from app import create_app
from app.extensions import db
from utility.db_setup.seed import Seed

# logging settings - hiding unnecessary third-party INFO logs
logger = test_logger

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
    return RagSettings() # the default settings (production)


@pytest.fixture(scope="session")
def vqa(settings):
    return VQA(settings) # the default settings (production)


@pytest.fixture(scope="function")
def query_data(rag_db):
    query = "Is there any skyscraper in the images?"
    rag_db, img_db, text_db, caption_db, client = rag_db
    query_results = img_db.query(query_texts=query,
                                n_results = BASELINE_K_SAMPLES,
                                include=["documents", "uris", "distances", "metadatas"])
    return query, query_results


def test_generate_visual_question_answers_01(vqa, query_data):
    query, query_results = query_data
    vq_answers = vqa.generate_visual_question_answers(query, query_results)
    
    # logging
    logger.info(f"VQAnswers:\n{vq_answers}")

    assert isinstance(vq_answers, list)
    assert isinstance(vq_answers[0], str)