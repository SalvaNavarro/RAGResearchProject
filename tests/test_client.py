import pytest
from services.rag.client import DBClient
from services.rag.rag_db import RagDB
from config import test_logger
from app import create_app
from app.extensions import db
from utility.db_setup.seed import Seed

logger = test_logger
modes = ["production", "test"]


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


@pytest.mark.parametrize("mode", modes)
def test_DBClient_01(rag_db, mode):
    """Checks client does access db by fetching collections"""
    client_obj = DBClient(mode=mode)
    col = client_obj.client.list_collections()
    col_names = [c.name for c in col]

    # logging
    logger.info(f"Collections on {mode} DB:\n{col}")

    assert "image" in col_names
    assert "caption" in col_names
    assert "text" in col_names