import pytest
from services.rag.rag_db import RagDB
from services.rag.client import DBClient
from services.rag.caption_generator import CaptionGenerator
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



def test_caption_generator_01(app_db, rag_db):
    """Checks caption collection is populated"""
    db, img_db, text_db, caption_db, client = rag_db
    cap_gen = CaptionGenerator()
    
    # deleting the captions collection so it can be re-created for the test
    caps = caption_db.get()
    caps_ids = [ci for ci in caps["ids"]]
    caption_db.delete(ids=caps_ids)

    # logging before adding captions
    logger.info(f"BEFORE:"
                f"\nText DB count = {text_db.count()}"
                f"\nImage DB count: {img_db.count()}"
                f"\nCaption DB count: {caption_db.count()}")

    # add captions to db
    cap_gen.add_captions_to_rag_db(img_db, caption_db)

    # logging after adding captions
    logger.info(f"AFTER:"
                f"\nText DB count = {text_db.count()}"
                f"\nImage DB count: {img_db.count()}"
                f"\nCaption DB count: {caption_db.count()}")

    assert caption_db.count() > 0