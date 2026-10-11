import pytest

from config import test_logger
from utility.rag_evaluation.rag_config import BASELINE_RERANKER_MODEL
from services.rag.reranker import Reranker

logger = test_logger

@pytest.fixture(scope="session")
def reranker():
    return Reranker(BASELINE_RERANKER_MODEL)


def test_rerank_01(reranker):
    question = "Are there any skyscrapers?"
    top_k = 3

    # creating dummy inputs
    docs = ["skyscrapers are present", 
            "a grass patch", 
            "a living room with large windows", 
            "dozens of cars", 
            "urban design considerations"]
    ids = ["top_id1", "top_id2", "top_id3", "top_id4", "top_id5"]
    surfaced = zip(docs, ids)

    reranked = reranker.rerank(question, list(surfaced), top_k)

    # logging
    logger.info(f"Originally surfaced:\n{surfaced}")
    logger.info(f"Reranked:\n{reranked}")

    assert len(docs) >= len(reranked)
    assert len(reranked) == top_k and isinstance(reranked[0], tuple)
    assert isinstance(reranked[0][0], str)