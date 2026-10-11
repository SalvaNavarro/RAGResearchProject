from datasets import load_dataset
from utility.rag_evaluation.rag_config import BENCHMARKER_SET_SIZE

class Benchmarker:
    """
    Creates a vector database from other public datasets for benchmarking.

    Attributes:
        dataset_path (str)
        dataset_name (str)
        ids (list[list[str]])
        questions (list[list[str]])
        documents (list[list[str]])
        response (list[list[str]])
    """
    def __init__(self) -> None:
        """
        Constructor.

        Returns:
            None
        """
        # dataset used for bencharking: RAGBench, secondary data, public
        self.dataset_path: str = "galileo-ai/ragbench" # public on HuggingFace
        self.dataset_name: str = "techqa"

        full_ds = load_dataset(self.dataset_path, self.dataset_name)
        ds = full_ds["test"] # using only the test split
        
        size = BENCHMARKER_SET_SIZE
        self.ids: list[str] = [src for src in list(ds["id"])] [:size]
        self.sources: list[list[str]] = [[src] for src in list(ds["id"])] [:size]
        self.documents: list[str] = self._flatten_list(list(ds["documents"])) [:size]
        self.references: list[list[str]] = list(ds["documents"]) [:size]

        questions: list[str] = list(ds["question"])
        responses: list[str] = list(ds["response"])
        self.questions: list[tuple[str, str]] = list(zip(questions, responses)) [:size]


    @staticmethod
    def _flatten_list(docs: list[list[str]]) -> list[str]:
        """
        Flattens the initial list[list[str]].

        Returns:
            list[str]
        """
        flat_docs: list[str] = [doc for sublist in docs for doc in sublist ]
        return flat_docs


if __name__ == "__main__":

    b = Benchmarker()
    print("OK")
