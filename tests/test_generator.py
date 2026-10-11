import pytest

from config import test_logger
from services.rag.generator import Generator
from services.rag.settings import RagSettings

logger = test_logger

@pytest.fixture(scope="session")
def settings_01():

    return RagSettings() # the default settings (production)


@pytest.fixture(scope="session")
def generator(settings_01):
    return Generator(settings_01)


@pytest.fixture(scope="session")
def rewriter(settings_01):
    return Generator(settings_01, rewriting=True)


def test_answer_01(generator):
    """Checks an answer has been generated"""
    generator = generator
    question = "How many people were in the stadium?"
    context = "The stadium can seat up to 100,000 people."
    answer = generator.answer(question, context)

    # logging
    logger.info(f"Question:\n{question}")
    logger.info(f"Answer:\n{answer}")
    logger.info(f"Contexts:\n{context}")

    assert isinstance(answer, str)


def test_answer_02(rewriter):
    """Checks a rewritten query has been generated"""
    # query includes a typo
    initial_query = """What are the architectural interests,
                or the scientific interests of
                the population in Verlin, Germany?"""
    context = "Many people enjoy gardens and green spaces."
    rewritten_query = rewriter.answer(initial_query, context)

    # logging
    logger.info(f"Original query:\n{initial_query}")
    logger.info(f"Rewritten query:\n{rewritten_query}")

    assert rewritten_query != initial_query
    assert isinstance(rewritten_query, str)