from config import *
ignore_warnings()
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Using CrossEncoder, this class takes (query, doc) tuples and predicts
    a similarity score, which will be used to rerank the initially retrieved
    top_results.

    Attributes:
        model (CrossEncoder): A sentence transformer.
        batch_size (int): Parameter for model batch size.
    """

    def __init__(self, 
                 model_rr: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size_rr: int = 32) -> None:
        """
        Constructor.

        Agrs:
            model (str): The chosen CrossEncoder model name.
            batch_size (int): Batch size setting for model prediction.
        """

        # log activity
        rag_logger.info(f"Initialising {self.__class__.__name__}")
        
        # CrossEncoder takes 2 str as input and predicts a similarity score
        self.model = CrossEncoder(model_rr)
        self.batch_size = batch_size_rr

    def rerank(self,
               query: str,
               top_results: list[tuple[str, str]],
               top_k: int | None = None) -> list[tuple[str, str]]:
        """
        Reranks an initial list of top results.

        Args:
            query (str): The input question.
            top_results (list[tuple[str, str]]): Initial list of top results.
            top_k (int | None): Maximum returned reranked results.

        Returns:
            list[tuple[str, str]]
        """

        # getting the docs from top_results as a list[str]
        docs: list[str]
        docs, _ = zip(*top_results)

        # using a list of tuples of query + doc as input to model.predict()
        inputs: list[tuple[str, str]] = [(query, d) for d in docs]
        scores = self.model.predict(inputs, batch_size=self.batch_size)

        # sorting top_results based on the new scoring
        paired = list(zip(scores, top_results))
        paired.sort(key=lambda t: t[0], reverse=True)
        reranked = [result for _, result in paired]

        # slicing the list if input top_k is not None
        if top_k is not None:
            return reranked[:top_k]
        
        else:
            return reranked





