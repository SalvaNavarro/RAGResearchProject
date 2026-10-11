from config import *
ignore_warnings()
from services.rag.settings import RagSettings
from services.rag.settings_heavy import RagSettingsHeavy
from langchain_community.chat_models import ChatOllama
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipImageProcessor, AutoTokenizer

class VQA:
    """
    Visual question answering class that queries images 
    content based on input queries.

    Attributes:
        settings (RagSettings): The settings object.
        settings_h (RagSettingsHeavy): The settings object that loads torch.
        model (ChatOllama): The LLM.
        prompt (str): The prompt for visual question answering.
        vqa_processor (BlipProcessor)
        vqa_model (BlipForQuestionAnswering)

    """
    def __init__(self, settings: RagSettings, 
                 settings_h: RagSettingsHeavy=RagSettingsHeavy()):

        # log activity
        rag_logger.info(f"Initialising {self.__class__.__name__}")

        # hyperparameters
        self.settings: RagSettings = settings
        self.settings_h = settings_h

        self.prompt: str = self._set_prompt()

        # load core VQA logic
        img_processor = BlipImageProcessor.from_pretrained(self.settings_h.vqa_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.settings_h.vqa_model_id, use_fast=True)
        self.vqa_processor = BlipProcessor(img_processor, tokenizer)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(self.settings_h.vqa_model_id)
        self.vqa_model.to(self.settings_h.device, dtype=self.settings_h.dtype).eval()

        
    def generate_visual_question_answers(self, query: str, query_results) -> list[str]:
        """
        Helper method to generate answers from Chroma query results.

        Parameters:
            query_results (Any): The raw Chroma query.

        Returns:
            list[str]
        """
        
        images = []
        vq_answers = []

        # visual question answering
        uris = query_results['uris'][0] # note: 0 is due to the default returned shape for single queries

        for uri in uris:
            images.append((uri, Image.open(uri).convert("RGB")))

        # feed images to torch
        for uri, img in images:
            with torch.no_grad():
                prompt = f"Question: {query} Answer with one concise sentence."
                inputs = self.vqa_processor(images=img, text=prompt, return_tensors="pt").to(self.settings_h.device)
                out = self.vqa_model.generate(**inputs, 
                                         min_new_tokens=self.settings.min_new_tokens, 
                                         max_new_tokens=self.settings.max_new_tokens, 
                                         num_beams=self.settings.num_beams,               
                                         length_penalty=self.settings.length_penalty,        
                                         no_repeat_ngram_size=self.settings.no_repeat_ngram_size)    
                ans = self.vqa_processor.decode(out[0], skip_special_tokens=True).strip().lower()
                vq_answers.append(ans)

        return vq_answers


    def _set_prompt(self) -> None:
        """
        Sets the final answer's prompt template.

        Returns:
            str
        """

        prompt = """
        Read the below context:
        {context}

        Answer the question based only on the information in the above context: 
        {question}
        """

        return prompt