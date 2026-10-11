from config import *
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# torch and BLIP imports placed on this settings file
BASELINE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASELINE_DTYPE = torch.float16
BASELINE_VQA_MODEL_ID = "Salesforce/blip-vqa-base"
# "Salesforce/blip-vqa-capfilt-large" # errors due to vulnerability, unable to use
BASELINE_CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
BASELINE_CAP_PROCESSOR = BlipProcessor.from_pretrained(BASELINE_CAPTION_MODEL_ID)
BASELINE_CAP_MODEL = BlipForConditionalGeneration.from_pretrained(BASELINE_CAPTION_MODEL_ID).to(BASELINE_DEVICE).eval()


class RagSettingsHeavy:
    """
    "Heavy" settings class that loads torch and BLIP,
    Only used for captions and VQA.

    """

    def __init__(self) -> None:
        """
        Constructor.

        Returns:
            None

        """
        self.device: str= BASELINE_DEVICE
        self.dtype= BASELINE_DTYPE
        self.vqa_model_id: str = BASELINE_VQA_MODEL_ID
        self.caption_model_id: str=BASELINE_CAPTION_MODEL_ID 
        self.cap_processor: str= BASELINE_CAP_PROCESSOR
        self.cap_model = BASELINE_CAP_MODEL