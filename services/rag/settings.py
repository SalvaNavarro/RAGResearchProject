import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration
from config import *
from utility.rag_evaluation.rag_config import *


class RagSettings:
    """
    Settings class that will be passed on to the different RAG classes.

    """
    
    def __init__(self, 
                 setup: str = "production", params: dict | None = None,
                 ) -> None:
        """
        Constructor.

        Returns:
            None

        """

        if setup == "production":
             params = PRODUCTION_PARAMS

        try:
            self.rag_type = params["rag_type"]

            self.model_choice = params["model_choice"]
            self.text_embeddings_model = params["text_embeddings_model"]
            self.initial_k_samples = params["initial_k_samples"]
            self.k_samples = params["k_samples"]
            self.text_weight = params["text_weight"]
            self.img_weight = params["img_weight"]
            self.caps_weight = params["caps_weight"]

            # torch imports make no performance difference
            # however, these have been moved to settings heavy class
            # self.device = params["device"]
            # self.dtype = params["dtype"]

            self.min_new_tokens = params["min_new_tokens"]
            self.max_new_tokens = params["max_new_tokens"]
            self.num_beams = params["num_beams"]
            self.length_penalty = params["length_penalty"]
            self.no_repeat_ngram_size = params["no_repeat_ngram_size"]
            
            self.pre_compute_captions = params["pre_compute_captions"]

            # BLIP imports do greatly reduce performance
            # these have been moved to SettingsHeavy class
            # self.caption_model_id = params["caption_model_id"]
            # self.cap_processor = params["cap_processor"]
            # self.cap_model = params["cap_model"]

            self.use_reranker = params["use_reranker"]
            self.reranker_model = params["reranker_model"]
            self.reranker_batch_size = params["reranker_batch_size"]

            self.use_rewriter = params["use_rewriter"]
            self.rewriter_model = params["rewriter_model"]
            self.domain_hints = params["domain_hints"]

        except Exception as e:
                print("Please pass parameters from rag_config.py as argument")