# generic
import pandas as pd
import evaluate
import asyncio
from statistics import mean

# ragas
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ResponseGroundedness # the NVIDIA groundedness metric from ragas
from ragas.metrics import ResponseRelevancy # the RAGAS response relevancy metric from ragas -> equivalent to "Answer Relevance"
from ragas.metrics import ContextRelevance # the NVIDIA context relevance metric from ragas
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# langchain
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama # changed this during RAGAS troubleshooting

# project
from config import RAG_RETRIEVAL_EVAL_CSV
from utility.rag_evaluation.rag_config import (BASELINE_K_SAMPLES, 
                                               RAG_EXPECTED_REFERENCES)
from utility.rag_evaluation.evaluator import Evaluator
from utility.rag_evaluation.benchmarker import Benchmarker


class Metrics:

    def __init__(self, eval: Evaluator, mode: str | None=None) -> None:
        # metric_names_ret = ["Precision@K", "Recall@K", "MRR"]
        metric_names_ret = ["Hit@K", "MRR"]
        metric_names_gen = ["BLEU", "ROUGE", "METEOR"]
        metric_names_hol = ["Groundedness", "Answer_Relevance", "Context_Relevance"]
        metric_names_time = ["Avg_Time(s)", "Time_To_Complete(s)"]
        hol_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # hol_llm = Ollama(model="llama3.2")
        hol_llm = ChatOllama(model="mistral:latest", temperature=0, top_p=1, seed=0, format="json")
        # hol_llm = raw_llm.bind(format="json")
        # constants
        if mode == "benchmarking":
            bm = Benchmarker()
            self.expected_references = bm.references

        else:
            self.expected_references = RAG_EXPECTED_REFERENCES


        # utility attributes
        self.eval: Evaluator = eval
        self.eval.run_evaluation()
        self.questions_count = len(self.eval.questions)
        # holistic metrics attributes
        self.hol_samples: list[SingleTurnSample] = self._create_samples()
        self.hol_llm = LangchainLLMWrapper(hol_llm)
        self.hol_embeddings = LangchainEmbeddingsWrapper(hol_embeddings)
        # output metrics attributes
        self.metrics_retrieval = pd.DataFrame(columns=metric_names_ret)
        self.metrics_generation = pd.DataFrame(columns=metric_names_gen)
        self.metrics_holistic = pd.DataFrame(columns=metric_names_hol)
        self.metrics_time = pd.DataFrame(columns=metric_names_time)
        self.metrics = pd.DataFrame()

        # run evaluation
        self._aggregate_values()


    def _extract_column(self, col) -> pd.Series:
        df = self._read_csv()
        col = df[col]
        return col

    
    @staticmethod
    def _read_csv() -> pd.DataFrame:
        df = pd.read_csv(RAG_RETRIEVAL_EVAL_CSV)
        return df
    
    # RETRIEVAL METRICS
    # Precision (generic)
    def _calculate_precision(self):
        values = self._extract_column("result")
        true_positives = values.sum()
        total = len(values)

        if total == 0:
            return 0.0
        
        # precision: TP / (TP + FP)
        false_positives = total - true_positives
        precision_long = true_positives / (true_positives + false_positives)
        precision = round(precision_long, 4)

        return precision

    # Recall (generic)
    @staticmethod
    def _calculate_recall(true_positives, false_negatives: int):

        # recall@K formula: TP / (TP + FN)
        if true_positives + false_negatives == 0:
            recall = 0
        else:
            r = true_positives / (true_positives + false_negatives)
            recall = round(r, 4)

        return recall
    
    # Precision@K
    def _calculate_precision_k(self, true_positives: int):
        # precision@K: TP / (TP + FP) * TOTAL
        p = true_positives / (BASELINE_K_SAMPLES * self.questions_count) # dividing true positives by total retrieved
        precision_k = round(p, 4)

        return precision_k

    # Recall@K
    @staticmethod
    def _calculate_recall_k(true_positives, false_negatives: int):
        # recall@K formula: TP / (TP + FN) * TOTAL
        if true_positives + false_negatives == 0:
            recall_k = 0
        else:
            r = true_positives / (true_positives + false_negatives)
            recall_k = round(r, 4)

        return recall_k

        
    # Hit@K -> alternative method where I only have 1 gold source / reference
    @staticmethod
    def _calculate_hit_k(true_positives):
        # recall@K formula: any TP within the top K
        if true_positives == 0:
            hit_k = 0.0
        else:
            hit_k = 1.0

        return hit_k

    
    # Mean Reciprocal Rank (MRR)
    @staticmethod
    def _calculate_mean_reciprocal_rank(ranks: list[int]) -> float:
        # ranks is the list of first-relevant position for each query
        reciprocal_ranks = []

        for rank in ranks:
            if rank > 0:
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0.0)

        mrr = sum(reciprocal_ranks) / len(ranks)
        mean_reciprocal_ranks = round(mrr, 4)

        return mean_reciprocal_ranks


    # GENERATION METRICS
    # BLEU
    def _calculate_bleu(self):
        eval_bleu = evaluate.load("bleu")
        bleu = eval_bleu.compute(predictions=self.eval.answers_actual, 
                                 references=self.expected_references[:self.questions_count],
                                 smooth=True,
                                 max_order=2)
        return round(bleu['bleu'], 4)

    # ROUGE
    def _calculate_rouge(self):
        eval_rouge = evaluate.load("rouge")
        rouge = eval_rouge.compute(predictions=self.eval.answers_actual, 
                                   references=self.expected_references[:self.questions_count])
        return round(rouge['rouge1'], 4)
    
    # METEOR
    def _calculate_meteor(self):
        eval_meteor = evaluate.load("meteor")
        meteor = eval_meteor.compute(predictions=self.eval.answers_actual, 
                                     references=self.expected_references[:self.questions_count])
        return round(meteor['meteor'], 4)


    # HOLISTIC METRICS -> async functions required for RAGAS metrics
    # groundedness
    async def _calculate_groundedness(self) -> float:
        scorer = ResponseGroundedness(llm=self.hol_llm)
        scores = []

        for sample in self.hol_samples:
            score = await scorer.single_turn_ascore(sample)
            scores.append(score) # this is an async function that needs an await statement
        
        return round(mean(scores), 4)

    # answer relevance
    async def _calculate_answer_relevance(self) -> list[float]:
        # Response Relevancy to measure answer relevance
        scores = []

        for sample in self.hol_samples:
            scorer = ResponseRelevancy(llm=self.hol_llm, embeddings=self.hol_embeddings)
            score = await scorer.single_turn_ascore(sample)
            scores.append(score)
    
        return round(mean(scores), 4)

    # context relevance
    async def _calculate_context_relevance(self) -> list[float]:
        scores = []

        for sample in self.hol_samples:
            scorer = ContextRelevance(llm=self.hol_llm)
            score = await scorer.single_turn_ascore(sample)
            scores.append(score)
        
        return round(mean(scores), 4)

    # create sample
    def _create_samples(self) -> SingleTurnSample:
        samples: list[SingleTurnSample] = []

        for (q, _), a, r in zip(self.eval.questions, self.eval.answers_actual, self.eval.contexts):
            sample = SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=r
            )
            samples.append(sample)

        return samples


    def _aggregate_values(self) -> None:
        # self.metrics.loc[0, "Precision"] = self._calculate_precision()
        # self.metrics.loc[0, "Recall"] = self._calculate_recall(self.eval_text.true_positive_count, self.eval_text.false_negative_count)
        # self.metrics_retrieval.loc[0, "Precision@K"] = self._calculate_precision_k(self.eval.true_positives)
        # self.metrics_retrieval.loc[0, "Recall@K"] = self._calculate_recall_k(self.eval.true_positives, self.eval.false_negatives)
        
        # replacing Precision@K with Hit@K, due to just 1 gold source on eval_config
        self.metrics_retrieval.loc[0, "Hit@K"] = self._calculate_hit_k(self.eval.true_positives)
        self.metrics_retrieval.loc[0, "MRR"] = self._calculate_mean_reciprocal_rank(self.eval.ranks)

        self.metrics_generation.loc[0, "BLEU"] = self._calculate_bleu()
        self.metrics_generation.loc[0, "ROUGE"] = self._calculate_rouge()
        self.metrics_generation.loc[0, "METEOR"] = self._calculate_meteor()

        self.metrics_holistic.loc[0, "Groundedness"] = asyncio.run(self._calculate_groundedness())
        self.metrics_holistic.loc[0, "Answer_Relevance"] = asyncio.run(self._calculate_answer_relevance())
        self.metrics_holistic.loc[0, "Context_Relevance"] = asyncio.run(self._calculate_context_relevance())

        self.metrics_time.loc[0, "Avg_Time(s)"] = round(float(self.eval.time_delta) / len(self.eval.questions), 2)
        self.metrics_time.loc[0, "Time_To_Complete(s)"] = round(float(self.eval.time_delta), 2)

        self.metrics = pd.concat([self.metrics_retrieval.reset_index(drop=True), 
                                  self.metrics_generation.reset_index(drop=True),
                                  self.metrics_holistic.reset_index(drop=True),
                                  self.metrics_time.reset_index(drop=True)],
                                  axis=1)