import pytest
from services.rag.rag_db import RagDB
from services.rag.client import DBClient
from config import test_logger
from app import create_app
from app.extensions import db
from utility.db_setup.seed import Seed

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


def test_rag_db_01(rag_db):
    """Checks DB is populated"""
    db, img_db, text_db, caption_db, client = rag_db

    # logging
    logger.info(f"\nText DB count = {text_db.count()}\nImage DB count: {img_db.count()}\nCaption DB count: {caption_db.count()}")
    
    assert text_db.count() > 0, img_db.count() > 0


def test_rag_db_02(rag_db):
    """Checks text chunks id format"""
    db, img_db, text_db, caption_db, client = rag_db
    sample_id = text_db.get(include=[])["ids"][0]

    # logging
    logger.info(f"Sample ID: {sample_id}")
    
    expected_id = "0_0"
    assert sample_id == expected_id


def test_rag_db_03(rag_db):
    """Checks image id format"""
    db, img_db, text_db, caption_db, client = rag_db
    sample_id = img_db.get(include=[])["ids"][0]

    # logging
    logger.info(f"Sample ID: {sample_id}")
    
    expected_id = "0_000-2-0.jpg"
    assert sample_id == expected_id


def test_rag_db_04(rag_db):
    """Checks text project id"""
    db, img_db, text_db, caption_db, client = rag_db
    results = text_db.get(ids=['0_0'], include=["metadatas"])
    # sample_src = results["metadatas"][0]["source"]
    project_id = results["metadatas"][0]["project_id"]

    expected_project_id = 0
    
    # logging
    logger.info(f"Sample project id: {project_id}")

    assert project_id == expected_project_id


def test_rag_db_05(rag_db):
    """Checks img project id"""
    db, img_db, text_db, caption_db, client = rag_db
    results = img_db.get(include=["metadatas"])
    # sample_uri = results["metadatas"][0]["uri"]
    sample_id = results["ids"][0]

    expected_id = "0_000-2-0.jpg"
    # expected_uri = 'app/static/projects/000_Project_000/000-2-0.jpg'
    
    # logging
    logger.info(f"Sample URI: {sample_id}")

    assert sample_id == expected_id