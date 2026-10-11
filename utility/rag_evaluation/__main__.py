import argparse
import pandas as pd
import optuna
from optuna.samplers import GridSampler
import os
from pathlib import Path
from functools import partial

from services.rag.settings import RagSettings
from utility.rag_evaluation.evaluator import Evaluator
from utility.rag_evaluation.metrics import Metrics
from utility.rag_evaluation.rag_config import TEXT, IMAGE, TEXT_IMAGE
from utility.rag_evaluation.rag_config import *
from utility.misc.util import add_zeroes

from config import *


def _objective(trial: optuna.Trial, mode: str, path: Path, setup: str,
               params_used: dict) -> float:
    """
    Optuna objective function sets the search space.

    Args:
        trial (optuna.Trial)
        mode (str): Production, test or benchmarking mode selection.
    
    Returns:
        float

    """

    if params_used is not None:

        # defining hyper parameters

        rag_type = trial.suggest_categorical("rag_type", params_used["rag_type"])
        model_choice = trial.suggest_categorical("model_choice", params_used["model_choice"])
        text_embeddings_model = trial.suggest_categorical("text_embeddings_model", params_used["text_embeddings_model"])

        initial_k_samples = trial.suggest_categorical("initial_k_samples", params_used["initial_k_samples"])
        k_samples = trial.suggest_categorical("k_samples", params_used["k_samples"])
        text_weight = trial.suggest_categorical("text_weight", params_used["text_weight"])
        img_weight = trial.suggest_categorical("img_weight", params_used["img_weight"])
        caps_weight = trial.suggest_categorical("caps_weight", params_used["caps_weight"])

        batch_size = trial.suggest_categorical("batch_size", params_used["batch_size"])
        min_new_tokens = trial.suggest_categorical("min_new_tokens", params_used["min_new_tokens"])
        max_new_tokens = trial.suggest_categorical("max_new_tokens", params_used["max_new_tokens"])
        decoding = trial.suggest_categorical("decoding", params_used["decoding"])
        num_beams = trial.suggest_categorical("num_beams", params_used["num_beams"])
        no_repeat_ngram_size = trial.suggest_categorical("no_repeat_ngram_size", params_used["no_repeat_ngram_size"])
        length_penalty = trial.suggest_categorical("length_penalty", params_used["length_penalty"])
        
        num_io_workers = trial.suggest_categorical("num_io_workers", params_used["num_io_workers"])
        cache_answers = trial.suggest_categorical("cache_answers", params_used["cache_answers"])
        
        pre_compute_captions = trial.suggest_categorical("pre_compute_captions", params_used["pre_compute_captions"])
        
        use_reranker = trial.suggest_categorical("use_reranker", params_used["use_reranker"]) 
        reranker_model = trial.suggest_categorical("reranker_model", params_used["reranker_model"])
        reranker_batch_size = trial.suggest_categorical("reranker_batch_size", params_used["reranker_batch_size"])
        
        use_rewriter = trial.suggest_categorical("use_rewriter", params_used["use_rewriter"])
        rewriter_model = trial.suggest_categorical("rewriter_model", params_used["rewriter_model"])
        domain_hints = trial.suggest_categorical("domain_hints", params_used["domain_hints"])

        params_used = {

            "rag_type" : rag_type,
            "model_choice" : model_choice,
            "text_embeddings_model" : text_embeddings_model,

            "initial_k_samples" : initial_k_samples,
            "k_samples" : k_samples, # final retrieved volume
            "text_weight" : text_weight, # text weighting
            "img_weight" : img_weight, # img weighting
            "caps_weight" : caps_weight, # caps weighting

            "batch_size" : batch_size,
            "min_new_tokens" : min_new_tokens,
            "max_new_tokens" : max_new_tokens,
            "decoding" : decoding,
            "num_beams" : num_beams,
            "no_repeat_ngram_size" : no_repeat_ngram_size,
            "length_penalty" : length_penalty,

            "num_io_workers" : num_io_workers,
            "cache_answers" : cache_answers,

            "pre_compute_captions" : pre_compute_captions,

            "use_reranker" : use_reranker,
            "reranker_model" : reranker_model,
            "reranker_batch_size" : reranker_batch_size,

            "use_rewriter" : use_rewriter,
            "rewriter_model" : rewriter_model,
            "domain_hints" : domain_hints

        }

    settings = RagSettings(setup=setup, params=params_used)
    
    # csv file name creation
    csv_metrics_path, iter_number = _create_path(path) 

    # log activity
    rag_logger.info(f"Starting iteration #{iter_number}...")

    # compute metrics and params
    metrics = _eval_single(settings, mode=mode)
    params_used = pd.DataFrame([params_used])

    # create dataframe and export csv for current iteration
    iteration = pd.concat([metrics.metrics, params_used], axis=1, ignore_index=False)
    iteration.to_csv(csv_metrics_path)

    # log activity
    rag_logger.info(f"Partial CSV exported to: {csv_metrics_path}")

    # reference scoring metric: groundedness
    score = metrics.metrics_holistic.loc[0, "Groundedness"]

    return score


def _create_path(path) -> None:
    """
    Creates an export file path and iteration number.

    Args:
        dir(Path): The directory where the csv files are.

    Returns:
        (Path, str)

    """

    i = 0
    i_formatted = add_zeroes(i) + str(i)

    while os.path.exists(path / f"metrics_set_{i_formatted}.csv" ):
        i += 1
        i_formatted = add_zeroes(i) + str(i)
    csv_metrics_path = path / f"metrics_set_{i_formatted}.csv"

    return csv_metrics_path, i_formatted


def _clear_previous_csv(dir: Path) -> None:
    """
    Clears folder to save the new set of csv files.

    Args:
        dir(Path): The directory where the csv files are.

    Returns:
        None

    """
    # remove previous set of files from directory
    if dir.is_dir():
        for file in dir.iterdir():
            file.unlink()


def _combine_csv(dir: Path) -> None:
    """
    Combines and exports all metrics into a single csv.

    """

    # initialise variables
    list_df: list[pd.DataFrame] = []
    file_path = dir / "all_combined_metrics.csv"

    # read all files and append to df list
    for file in dir.iterdir():
        df = pd.read_csv(file)
        list_df.append(df)

    # combine dataframes and export to csv
    combined = pd.concat(list_df, axis=0, ignore_index=False)
    combined.to_csv(file_path)

    # log activity
    rag_logger.info(f"Combined CSV exported to: {file_path}")


def _eval_single(settings: RagSettings, mode: str) -> pd.DataFrame:
    """
    Performs a single evaluation.

    Args:
        settings (RagSettings): The settings object.
        mode (str): Production, test or benchmarking mode selection.

    Returns:
        pd.DataFrame

    """

    eval = Evaluator(settings, mode)
    metrics = Metrics(eval, mode=mode)

    return metrics


def main(mode: str="production",) -> None:
    """
    Runs a performance evaluation under selected configurations.

    Args: 
        mode (str): Production, test or benchmarking mode selection.
        n_trials (int): Default number of trials set to 1, needs overridde.

    """

    # set mode
    if mode == "baseline":
        dir = RAG_BASELINE_EVAL_CSV
        path = RAG_BASELINE_EVAL_CSV
        setup = "baseline"
        params_used = BASELINE_PARAMS

    elif mode == "eval":
        dir = RAG_EVAL_CSV
        path = RAG_EVAL_CSV
        setup = "eval"

        # option 1: first pass
        # params_used = EVAL_PARAMS_FIRST_PASS

        # option 2: second pass
        params_used = EVAL_PARAMS_SECOND_PASS

    elif mode == "benchmarking":
        dir = RAG_BENCH_CSV
        path = RAG_BENCH_CSV
        setup = "benchmarking"
        params_used = {}

        # option 1: first pass
        # params_used = EVAL_PARAMS_FIRST_PASS
        # params_used["rag_type"] = [TEXT]
        # params_used["pre_compute_captions"] = [False]
        # params_used["domain_hints"] = [""]

        # option 2: second pass (very long time to compute)
        # params_used = EVAL_PARAMS_SECOND_PASS
        # params_used["rag_type"] = [TEXT]
        # params_used["pre_compute_captions"] = [False]
        # params_used["domain_hints"] = [""]

        # option 3: parsing dict values to list for parameter grid
        for key, value in zip(PRODUCTION_PARAMS.keys(), 
                              PRODUCTION_PARAMS.values()):
            params_used[key] = [value]

    # prepare grid
    sampler = GridSampler(params_used)

    # clear folder
    _clear_previous_csv(dir)

    # create study
    study = optuna.create_study(
        study_name="RAG Evaluation",
        direction="maximize",
        sampler=sampler)
    
    # using partial to wrap around objective and enable custom arguments
    custom_objective = partial(_objective, mode=mode, path=path, 
                               setup=setup, params_used=params_used)
    
    # launch optimisation
    study.optimize(custom_objective)

    # combine all metrics and export
    _combine_csv(dir)


if __name__ == "__main__":

    # arguments for running configurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="RAG evaluation mode.", 
                        default="eval")
    args = parser.parse_args()

    # run evaluation
    main(args.mode)
