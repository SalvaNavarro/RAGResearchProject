from config import *
ignore_warnings()
import numpy as np
from services.rag.settings import RagSettings
from chromadb import Collection


class Retriever:
    """
    Retriever class that accesses the database and returns relevant references
    from input queries.

    Attributes:
        settings (RagSettings): The settings object.
        rag_type (str): The type of rag (text/image/text_image)
        text_db (Collection): The text vector db collection.
        image_db (Collection): The image vector db collection.
        caption_db (Collection): The captions vector db collection.

    """

    def __init__(self, settings: RagSettings, 
                 text_db: Collection, image_db: Collection,
                 caption_db: Collection) -> None:
        """
        Constructor.

        Args:
            settings (RagSettings): The settings object.
            rag_type (str): The type of rag (text/image/text_image)
            text_db (Collection): The text vector db collection.
            image_db (Collection): The image vector db collection.
            caption_db (Collection): The captions vector db collection.

        Returns:
            None

        """
        # log activity
        rag_logger.info(f"Initialising {self.__class__.__name__}")  

        # hyperparameters
        self.settings = settings
        if self.settings.use_reranker:
            self.initial_k_samples = self.settings.initial_k_samples
        else:
            self.initial_k_samples = self.settings.k_samples
        # other attributes
        self.rag_type: str = self.settings.rag_type
        self.text_db: Collection = text_db
        self.image_db: Collection = image_db
        self.caption_db: Collection = caption_db


    def retrieve_from_query(self, query: str) -> tuple[str, list[str]]:
        """
        Retrieves sources and contexts from an input query. Handles access to
        relevant db according to self.rag_type.

        Args:
            query (str): The input query text.

        Returns:
            tuple[str, list[str]]

        """
        # merging all surfaced references
        top_results: list[tuple[str, str]]

        if self.rag_type == "text":
            top_results = self._retrieve_from_text_db(query, self.settings.pre_compute_captions)

        elif self.rag_type == "image":
            from services.rag.vqa import VQA
            self.vqa = VQA(self.settings) # to be used only on image or multimodal setups
            top_results = self._retrieve_from_image_db(query, self.settings.pre_compute_captions)
        
        elif self.rag_type == "text_image":
            from services.rag.vqa import VQA
            self.vqa = VQA(self.settings) # to be used only on image or multimodal setups
            top_results = self._retrieve_from_text_image_db(query, self.settings.pre_compute_captions)

        # reranker if enabled
        if self.settings.use_reranker:
            from services.rag.reranker import Reranker
            self.reranker = Reranker(model_rr=self.settings.reranker_model,
                                     batch_size_rr=self.settings.reranker_batch_size)

            top_results = self.reranker.rerank(query, list(top_results), top_k=self.settings.k_samples)

        else:
            top_results = list(top_results)[:self.settings.k_samples]

        # processing results
        contexts = []
        docs, ids = map(list, zip(*top_results))

        for doc in docs:
            contexts.append(doc)

        sources = []
        for id in ids:
            sources.append(id)

        return sources, contexts
    
    
    def _retrieve_from_text_db(self, query, 
                               use_captions) -> list[tuple[str, str]]: # review this type hinting
        """
        Retrieves sources/contexts from text collection from an input query.

        Args:
            query (str): The input query text.
            use_captions (bool): Flag to enable use of captions.

        Returns:
            list[tuple[str, str]]

        """
        
        surfaced_texts = self.text_db.query(query_texts=query,
                                            n_results=self.initial_k_samples,
                                            include=["documents", "uris", 
                                                     "distances", "metadatas"])

        top_dists_txt = surfaced_texts['distances'][0]
        top_docs_txt = surfaced_texts['documents'][0]
        top_ids_txt = surfaced_texts['ids'][0]

        # new block to add the caption logic
        if use_captions:
            all_surfaced_sorted = self._use_captions(query, 
                                                     top_docs_txt, 
                                                     top_ids_txt, 
                                                     top_dists_txt)

        else:
            all_surfaced_sorted = [(doc, id) 
                                   for _, doc, id in sorted(
                                       zip(top_dists_txt, 
                                           top_docs_txt, 
                                           top_ids_txt)
                                    )]

        return all_surfaced_sorted
    

    def _retrieve_from_image_db(self, query, 
                                use_captions) -> list[tuple[str, str]]:
        """
        Retrieves sources/contexts from image collection from an input query.

        Args:
            query (str): The input query text.
            use_captions (bool): Flag to enable use of captions.

        Returns:
            list[tuple[str, str]]

        """
        
        surfaced_images = self.image_db.query(query_texts=query,
                            n_results = self.initial_k_samples,
                            include=["documents", "uris", 
                                     "distances", "metadatas"])
        top_dists_imgs = surfaced_images['distances'][0]
        top_ids_imgs = surfaced_images['ids'][0]
        top_docs_imgs = self.vqa.generate_visual_question_answers(query, surfaced_images)

        # new block to add the caption logic
        if use_captions:
            all_surfaced_sorted = self._use_captions(query, 
                                                     top_docs_imgs, 
                                                     top_ids_imgs, 
                                                     top_dists_imgs)

        else:
            all_surfaced_sorted = [(doc, id) 
                                   for _, doc, id in sorted(
                                       zip(top_dists_imgs,
                                           top_docs_imgs, 
                                           top_ids_imgs)
                                    )]  

        return all_surfaced_sorted[:self.initial_k_samples]
    

    def _retrieve_from_text_image_db(self, question, 
                                     use_captions) -> list[tuple[str, str]]:
        
        """
        Retrieves sources/contexts from text and
        image collection from an input query.

        Args:
            query (str): The input query text.
            use_captions (bool): Flag to enable use of captions.

        Returns:
            list[tuple[str, str]]

        """
        
        # searching on text db
        surfaced_texts = self.text_db.query(query_texts=question,
                                            n_results=self.initial_k_samples,
                                            include=["documents", "uris", 
                                                     "distances", "metadatas"])

        top_docs_texts = surfaced_texts['documents'][0]
        top_dists_texts = surfaced_texts['distances'][0]
        top_ids_texts = surfaced_texts['ids'][0]

        # searching on image db
        surfaced_images = self.image_db.query(query_texts=question,
                            n_results = self.initial_k_samples,
                            include=["documents", "uris", "distances", "metadatas"])
        
        top_dists_imgs = surfaced_images['distances'][0]
        top_ids_imgs = surfaced_images['ids'][0]

        top_docs_imgs = self.vqa.generate_visual_question_answers(question, surfaced_images) # generate vqa

        # combining results
        if use_captions:
            top_docs_combined = top_docs_texts + top_docs_imgs
            top_dists_combined = top_dists_texts + top_dists_imgs
            top_ids_combined= top_ids_texts + top_ids_imgs
            # shortening list to just K items
            top_reduced = [(dist, doc, id) 
                           for dist, doc, id in sorted(zip(
                               top_dists_combined, 
                               top_docs_combined, 
                               top_ids_combined))]
            
            top_dists_other, top_docs_other, top_ids_other = map(list, zip(*top_reduced))
            all_surfaced_sorted = self._use_captions(question, 
                                                     top_docs_other[:self.initial_k_samples], 
                                                     top_ids_other[:self.initial_k_samples], 
                                                     top_dists_other[:self.initial_k_samples])

        else:
            top_docs = top_docs_texts + top_docs_imgs
            top_dists = self._combine_similarities(top_dists_texts, top_dists_imgs)
            top_ids = top_ids_texts + top_ids_imgs
            all_surfaced_sorted = [(doc, id) 
                                   for _, doc, id in sorted(zip(
                                       top_dists, top_docs, top_ids))]

        return all_surfaced_sorted[:self.initial_k_samples]
    

    def _combine_similarities(self, 
                              other_dists: list[float] | None = None,
                              text_dists: list[float] | None = None, 
                              img_dists: list[float] | None = None, 
                              caps_dists: list[float] | None = None) -> list[float]:
        """
        Helper method to combine the similarity scores for text/image/caption
        retrievals.

        Args:
            text_dists (list[float]): Similarities (distances) list from text.
            img_dists (list[float]): Similarities (distances) list from images.
            caps_dists (list[float]): Similarities (distances) list from caps.

        Returns:
            list[float]

        """
        text_w = self.settings.text_weight
        img_w = self.settings.img_weight
        caps_w = self.settings.caps_weight
        others_w = (text_w + img_w) / 2

        if text_dists is not None:
            # convert text distances to similarity
            t_sims = np.array(self._convert_to_normalised_similarity(text_dists))
            tw = text_w * t_sims
        else:
            tw = 0.0

        if img_dists is not None:
            # convert image distances to similarity
            img_sims = np.array(self._convert_to_normalised_similarity(img_dists))
            iw = img_w * img_sims
        else:
            iw = 0.0

        if caps_dists is not None:
            caps_sims = np.array(self._convert_to_normalised_similarity(caps_dists))
            cw = caps_w * caps_sims
        else:
            cw = 0.0

        # other_dists is a combination of text and image dists
        if other_dists is not None:
            other_sims = np.array(self._convert_to_normalised_similarity(other_dists))
            ow = others_w * other_sims
        else:
            ow = 0.0

        # weighted combined similarities as list
        combined_sims = tw + iw + cw + ow
        combined_sims = combined_sims.tolist()

        return combined_sims
    

    def _use_captions(self, query: str, top_docs_other, 
                      top_ids_other, top_dists_other) -> tuple[str, str]:
        """
        Helper method to add pre-computed captions to a set of documents 
        from text or image (VQA) collections.

        Args:
            top_docs_other (list[str]): Documents list.
            top_ids_other (list[str]): Ids list.
            top_dists_other (list[float]): Similarities (distances) list.

        Returns:
            list[float]

        """
        surfaced_caps = self.caption_db.query(query_texts=query,
                                n_results=self.initial_k_samples,
                                include=["documents", "uris", "distances", "metadatas"])

        top_dists_caps = surfaced_caps['distances'][0]
        top_ids_caps = surfaced_caps['ids'][0]
        top_docs_caps = surfaced_caps['documents'][0]

        top_docs = top_docs_other + top_docs_caps
        # depending on the question, similarities might only pick top docs from VQA or mix them up with captions
        top_dists = self._combine_similarities(top_dists_other, top_dists_caps)
        top_ids = top_ids_other + top_ids_caps
        all_surfaced_sorted = [(doc, id) for _, doc, id in sorted(zip(top_dists, top_docs, top_ids))]  

        return all_surfaced_sorted


    @staticmethod
    def _convert_to_normalised_similarity(distances: list[float]) -> list[float]:
        """
        Normalises the similarity score for correct weighting.

        Args:
            distances (list[float]): Similarities (distances) list.

        Returns:
            list[float]

        """
        sims = []
        norm_sims = []

        for dist in distances:
            sim = 1 / (1 + dist)
            sims.append(sim)

        norm_sims = []
        for sim in sims:
            norm_sim = (sim - min(sims)) / (max(sims) - min(sims))
            norm_sims.append(norm_sim)

        return norm_sims