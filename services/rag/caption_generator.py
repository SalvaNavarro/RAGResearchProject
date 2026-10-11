from config import *
ignore_warnings()
from PIL import Image
from services.rag.settings import RagSettings
from chromadb import Collection

from services.rag.settings_heavy import RagSettingsHeavy


class CaptionGenerator:
    """
    Generates captions from images that will be used by RagDB to populate
    the caption_db collection on the vector database.

    Attributes:
        settings (RagSettings): The settings for the rewriter models.
    """

    def __init__(self, settings: RagSettings=RagSettings(), settings_h: RagSettingsHeavy=RagSettingsHeavy()) -> None:
        """
        Constructor.

        Args:
            settings (RagSettings): The settings object.

        Returns:
            None
        """
                
        # log activity
        rag_logger.info(f"Initialising {self.__class__.__name__}")

        self.settings: RagSettings = settings # hyperparameter input
        self.settings_h: RagSettingsHeavy = settings_h


    def add_captions_to_rag_db(self, 
                               image_db: Collection, 
                               caption_db: Collection) -> None:
        """
        Adds the image captions to the rag db.

        Returns:
            None
        """

        # getting all entries from image collection
        entries: dict[str, list | None] = image_db.get(include=["uris"])
        img_ids: list[str] = entries["ids"]
        img_uris: list[str] = entries["uris"]

        # initialising relevant lists
        cap_ids = []
        cap_docs = []
        cap_metas = []

        # iterating through the retrieved image entries
        for img_id, uri in zip(img_ids, img_uris):

            # open each image using PIL and convert to RGB
            pil = Image.open(uri).convert("RGB")
            # creating a general caption
            caption = self._generate_caption(pil)
            # appending the ids, caption and metadatas to lists
            cap_ids.append(img_id)
            cap_docs.append(caption)
            cap_metas.append({
                "img_id": img_id,
                "uri": uri,
                "caption_general": caption
            })

            # further to the general caption, dense captions from crops could be added in the future
            # dense_captions = []
            # image cropping logic...

        # adding the captions to the caption_db collection
        caption_db.add(ids=img_ids, documents=cap_docs, metadatas=cap_metas)


    def _generate_caption(self, img: Image.Image) -> str:
        """
        Generates the actual text on the captions.

        Returns:
            str
        """

        # setting up caption processor model settings, returning Pytorch tensors
        inputs = self.settings_h.cap_processor(images=img, return_tensors="pt")
        # moves inputs to current device
        inputs = inputs.to(self.settings_h.device)
        # generating the caption
        out = self.settings_h.cap_model.generate(**inputs, 
                                               max_new_tokens=self.settings.max_new_tokens, 
                                               num_beams=self.settings.num_beams)
        # cleaning up the caption for output
        caption = self.settings_h.cap_processor.decode(out[0], skip_special_tokens=True).strip()

        return caption