from typing import Any
import pandas as pd
import time

from services.rag.query import QueryRag
from utility.rag_evaluation.rag_config import *
from config import *
from services.rag.settings import RagSettings


class Evaluator:
    """
    Evaluates the performance of one RAG setup  at a time.

    Attributes:
        questions list[tuple[str, str]]: the list of evaluation questions
        expected_sources list[list[str]]: the IDs of the expected sources
        rag (QueryRag): the query RAG object
        ... -> rest of attributes: expected answers, sources and metrics
    """
    def __init__(self, settings: RagSettings, mode="production") -> None:
        """
        Constructor.
        """

        # hyperparameters
        self.settings = settings

        # query rag
        self.rag: QueryRag = QueryRag(self.settings, mode=mode)

        # basic data attributes
        self.answers_actual: list[str] = []
        self.answers_expected: list[str] = []
        self.sources_actual: list[list[str]] = []
        self.contexts: list[list[str]] = []
        self.accuracies: list[bool] = [] # a simple list to broadly determine accuracy

        # results attributes
        self.partial_results: list[tuple[str, str, bool]] = []
        self.results: pd.DataFrame = pd.DataFrame()

        # retrieval-specific attributes
        self.true_positives_list: list[int] = []
        self.true_positives: int = 0
        self.false_negatives_list: list[int] = []
        self.false_negatives: int = 0
        self.ranks: list[int] = []

        # generation-specific attributes
        self.bleu: dict[Any, Any] = {}
        self.rouge: dict[Any, Any] = {}
        self.meteor: dict[Any, Any] = {}

        # time to complete task (average)
        self.time_delta: float = 0

        # mode
        if mode == "test":
            self.questions: list[tuple[str, str]] = RAG_QUESTION_LIST[:1]
            self.expected_sources: list[list[str]] = RAG_EXPECTED_SOURCES[:1]

        # benchmarking mode to compare with a public dataset of choice
        elif mode == "benchmarking":
            from utility.rag_evaluation.benchmarker import Benchmarker
            bm = Benchmarker()
            self.questions: list[tuple[str, str]] = bm.questions
            self.expected_sources: list[list[str]] = bm.sources
            # self.contexts: list[list[str]] = bm.references

        else:
            self.questions: list[tuple[str, str]] = RAG_QUESTION_LIST
            self.expected_sources: list[list[str]] = RAG_EXPECTED_SOURCES


    def run_evaluation(self) -> None:
        """
        Consolidate results into self.results.

        Returns:
            None
        """
        begin = time.perf_counter()

        for idx, (question, expected_answer) in enumerate(self.questions):
            result: bool = self.evaluate(idx, question, expected_answer)
            self.partial_results.append((question, expected_answer, result))

        end = time.perf_counter()
        self.time_delta = round((end - begin), 2)

        self.true_positives = sum(self.true_positives_list)
        self.false_negatives = sum(self.false_negatives_list)

        # convert to pandas data frame
        part_results = pd.DataFrame(self.partial_results).T
        src_actual = pd.DataFrame(self.sources_actual).T

        # combine results with sources
        self.results = pd.concat([part_results, src_actual], ignore_index=False).T
        # self.results.columns = self.col_names


    def evaluate(self, idx, question: str, expected_answer: str) -> bool:
        # receive query output
        response_with_src: tuple[str, list[str], list[str]] = self.rag.query(question)
        answer: str = response_with_src[0]
        sources: list[str] = response_with_src[1]
        contexts: list[str] = response_with_src[2]

        # add answers, TPs, FNs, ranks to appropriate lists
        tps = self.check_expected_sources(idx, sources)[0]
        fns = self.check_expected_sources(idx, sources)[1]
        self.answers_actual.append(answer)
        self.true_positives_list.append(tps)
        self.false_negatives_list.append(fns)
        self.ranks.append(self.calculate_rank())

        # append new sublist of sources to self.sources_actual
        self.sources_actual.append(sources)
        self.contexts.append(contexts)

        # prompting the model
        prompt = EVALUATION_PROMPT.format(expected_answer=expected_answer, actual_answer=answer)
        generator = self.rag.generator
        results_str = generator.model.invoke(prompt).content
        results_str_clean = results_str.strip().lower()

        # evaluation prompt already requests a 'true' or 'false' answer, thus the below logic:
        if "true" in results_str_clean:
            return True
        else:
            return False
        

    def check_expected_sources(self, idx, sources: list[str]) -> int:
        """
        Checks and add true positives / false negatives to the relevant lists.

        Returns: tuple[int, int]
        """
        # counters and lists
        true_positives = 0
        false_negatives = 0
        flat_src_list = []
        flat_src_list_extended = []


        # calculating true positives
        for src in self.expected_sources[idx]:
            flat_src_list.append(src)

        for src in sources:
            if any(src in full for full in flat_src_list):
                true_positives += 1

        # calculating false negatives (sources that should have been there)
        for src in self.expected_sources[idx]:
            flat_src_list_extended.append(src)

        for src in sources:
            if not any(src in full for full in flat_src_list):
                false_negatives += 1

        return true_positives, false_negatives


    def calculate_rank(self) -> int:
        # defaulting rank to a very high number, so returned rank value is very low if no matches
        rank: int = 100
        
        for idx, tp in enumerate(self.true_positives_list):
            if tp > 0:
                rank = idx + 1 # adding one to make the rank 1-based (so first item is #1)

        return rank