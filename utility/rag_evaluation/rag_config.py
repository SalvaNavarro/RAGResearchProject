__all__ = [
    "BASELINE_PARAMS",
    "EVAL_PARAMS_SECOND_PASS",
    "EVAL_PARAMS_FIRST_PASS",
    "PRODUCTION_PARAMS",
    "EVALUATION_PROMPT",
    "RAG_QUESTION_LIST",
    "RAG_EXPECTED_SOURCES",
    "RAG_EXPECTED_REFERENCES",
    "BASELINE_RAG_TYPE",
    "BASELINE_LLM",
    "BASELINE_INITIAL_K_SAMPLES",
    "BASELINE_K_SAMPLES",
    "BASELINE_TEXT_WEIGHT",
    "BASELINE_IMG_WEIGHT",
    "BASELINE_CAPS_WEIGHT",
    "BASELINE_TEXT_EMBEDDING_MODEL",
    "BASELINE_BATCH_SIZE",
    "BASELINE_MAX_NEW_TOKENS",
    "BASELINE_MIN_NEW_TOKENS",
    "BASELINE_DECODING",
    "BASELINE_NUM_BEAMS",
    "BASELINE_NO_REPEAT_NGRAM_SIZE",
    "BASELINE_LENGTH_PENALTY",
    "BASELINE_NUM_IO_WORKERS",
    "BASELINE_CACHE_ANSWERS",
    "BASELINE_CAPTION_FLAG",
    "BASELINE_USE_RERANKER",
    "BASELINE_RERANKER_MODEL",
    "BASELINE_RERANKER_BATCH_SIZE",
    "BASELINE_USE_REWRITER",
    "BASELINE_REWRITER_MODEL",
    "BASELINE_DOMAIN_HINTS"
]


# rag types
TEXT = "text"
IMAGE = "image"
TEXT_IMAGE = "text_image"

# BASELINE PARAMETERS
# rag type
BASELINE_RAG_TYPE = TEXT

# LLM model parameters
BASELINE_LLM = "llama3.2:latest"
BASELINE_TEXT_EMBEDDING_MODEL = "nomic-embed-text"

# general retrieval and weighting settings
BASELINE_INITIAL_K_SAMPLES = 18 # preliminary K_Samples before reranking
BASELINE_K_SAMPLES = 9 # set to 9 for production to max out interface grid
BASELINE_TEXT_WEIGHT = 1.0
BASELINE_IMG_WEIGHT = 1.0
BASELINE_CAPS_WEIGHT = 1.0

# generation settings
BASELINE_BATCH_SIZE = 4
BASELINE_MAX_NEW_TOKENS = 60
BASELINE_MIN_NEW_TOKENS = 20
BASELINE_DECODING = "greedy"
BASELINE_NUM_BEAMS = 1
BASELINE_NO_REPEAT_NGRAM_SIZE = 3
BASELINE_LENGTH_PENALTY = 1.0

# performance
BASELINE_NUM_IO_WORKERS = 8
BASELINE_CACHE_ANSWERS = True

# captions flag
BASELINE_CAPTION_FLAG = True

# reranker flag & settings
BASELINE_USE_RERANKER = False
BASELINE_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BASELINE_RERANKER_BATCH_SIZE = 32

# rewriter flag, settings and domain hints
BASELINE_USE_REWRITER = False
BASELINE_REWRITER_MODEL = "zorowin123/rq_rag_llama2_7B"
BASELINE_DOMAIN_HINTS = ("architecture, architectural design, buildings,"
                         "building elements, building structures, public spaces, "
                         "facades, interior spaces.")


BASELINE_PARAMS = {

    "rag_type" : [TEXT, IMAGE, TEXT_IMAGE],
    "model_choice" : [BASELINE_LLM],
    "text_embeddings_model" : [BASELINE_TEXT_EMBEDDING_MODEL],

    "initial_k_samples" : [BASELINE_INITIAL_K_SAMPLES],
    "k_samples" : [BASELINE_K_SAMPLES], # final retrieved volume
    "text_weight" : [BASELINE_TEXT_WEIGHT], # text weighting
    "img_weight" : [BASELINE_IMG_WEIGHT], # img weighting
    "caps_weight" : [BASELINE_CAPS_WEIGHT], # caps weighting

    "batch_size" : [BASELINE_BATCH_SIZE],
    "min_new_tokens" : [BASELINE_MIN_NEW_TOKENS],
    "max_new_tokens" : [BASELINE_MAX_NEW_TOKENS],
    "decoding" : [BASELINE_DECODING],
    "num_beams" : [BASELINE_NUM_BEAMS],
    "no_repeat_ngram_size" : [BASELINE_NO_REPEAT_NGRAM_SIZE],
    "length_penalty" : [BASELINE_LENGTH_PENALTY],

    "num_io_workers" : [BASELINE_NUM_IO_WORKERS],
    "cache_answers" : [BASELINE_CACHE_ANSWERS],

    "pre_compute_captions" :[BASELINE_CAPTION_FLAG],

    "use_reranker" :[BASELINE_USE_RERANKER],
    "reranker_model" :[BASELINE_RERANKER_MODEL],
    "reranker_batch_size" : [BASELINE_RERANKER_BATCH_SIZE],

    "use_rewriter" : [BASELINE_USE_REWRITER],
    "rewriter_model" : [BASELINE_REWRITER_MODEL],
    "domain_hints" : [BASELINE_DOMAIN_HINTS]
}

# first pass: a coarse pass to show the main parameters
EVAL_PARAMS_FIRST_PASS = {

    "rag_type" : [TEXT, IMAGE, TEXT_IMAGE],
    "model_choice" : [BASELINE_LLM],
    "text_embeddings_model" : [BASELINE_TEXT_EMBEDDING_MODEL],

    "initial_k_samples" : [BASELINE_INITIAL_K_SAMPLES],
    "k_samples" : [BASELINE_K_SAMPLES], # final retrieved volume
    "text_weight" : [BASELINE_TEXT_WEIGHT], # text weighting
    "img_weight" : [BASELINE_IMG_WEIGHT], # img weighting
    "caps_weight" : [BASELINE_CAPS_WEIGHT], # caps weighting

    "batch_size" : [BASELINE_BATCH_SIZE],
    "min_new_tokens" : [BASELINE_MIN_NEW_TOKENS],
    "max_new_tokens" : [BASELINE_MAX_NEW_TOKENS],
    "decoding" : [BASELINE_DECODING],
    "num_beams" : [BASELINE_NUM_BEAMS],
    "no_repeat_ngram_size" : [BASELINE_NO_REPEAT_NGRAM_SIZE],
    "length_penalty" : [BASELINE_LENGTH_PENALTY],

    "num_io_workers" : [BASELINE_NUM_IO_WORKERS],
    "cache_answers" : [BASELINE_CACHE_ANSWERS],

    "pre_compute_captions" :[BASELINE_CAPTION_FLAG, False],

    "use_reranker" :[BASELINE_USE_RERANKER, True],
    "reranker_model" :[BASELINE_RERANKER_MODEL],
    "reranker_batch_size" : [BASELINE_RERANKER_BATCH_SIZE],

    "use_rewriter" : [BASELINE_USE_REWRITER, True],
    "rewriter_model" : [BASELINE_REWRITER_MODEL],
    "domain_hints" : [BASELINE_DOMAIN_HINTS]
}


# second pass: a fine-tuning pass to improve performance
EVAL_PARAMS_SECOND_PASS = {

    "rag_type" : [TEXT],
    "model_choice" : [BASELINE_LLM, "mistral:latest"],
    "text_embeddings_model" : [BASELINE_TEXT_EMBEDDING_MODEL],

    "initial_k_samples" : [BASELINE_INITIAL_K_SAMPLES],
    "k_samples" : [BASELINE_K_SAMPLES, 5], # final retrieved volume
    "text_weight" : [BASELINE_TEXT_WEIGHT], # text weighting
    "img_weight" : [BASELINE_IMG_WEIGHT], # img weighting
    "caps_weight" : [BASELINE_CAPS_WEIGHT], # caps weighting

    "batch_size" : [BASELINE_BATCH_SIZE, 2],
    "min_new_tokens" : [BASELINE_MIN_NEW_TOKENS, 50],
    "max_new_tokens" : [BASELINE_MAX_NEW_TOKENS, 300],
    "decoding" : [BASELINE_DECODING],
    "num_beams" : [BASELINE_NUM_BEAMS, 4],
    "no_repeat_ngram_size" : [BASELINE_NO_REPEAT_NGRAM_SIZE, 3],
    "length_penalty" : [BASELINE_LENGTH_PENALTY, 1.1],

    "num_io_workers" : [BASELINE_NUM_IO_WORKERS],
    "cache_answers" : [BASELINE_CACHE_ANSWERS],

    "pre_compute_captions" :[False],

    "use_reranker" :[BASELINE_USE_RERANKER],
    "reranker_model" :[BASELINE_RERANKER_MODEL],
    "reranker_batch_size" : [BASELINE_RERANKER_BATCH_SIZE],

    "use_rewriter" : [BASELINE_USE_REWRITER],
    "rewriter_model" : [BASELINE_REWRITER_MODEL],
    "domain_hints" : [BASELINE_DOMAIN_HINTS, ""] # testing with and without hints
}


# production parameters: a set of selected parameters for production use
PRODUCTION_PARAMS = {

    "rag_type" : TEXT,
    "model_choice" : BASELINE_LLM,
    "text_embeddings_model" : BASELINE_TEXT_EMBEDDING_MODEL,

    "initial_k_samples" : BASELINE_INITIAL_K_SAMPLES,
    "k_samples" : BASELINE_K_SAMPLES, # final retrieved volume
    "text_weight" : BASELINE_TEXT_WEIGHT, # text weighting
    "img_weight" : BASELINE_IMG_WEIGHT, # img weighting
    "caps_weight" : BASELINE_CAPS_WEIGHT, # caps weighting

    "batch_size" : 2,
    "min_new_tokens" : 50,
    "max_new_tokens" : BASELINE_MAX_NEW_TOKENS,
    "decoding" : BASELINE_DECODING,
    "num_beams" : 4,
    "no_repeat_ngram_size" : BASELINE_NO_REPEAT_NGRAM_SIZE,
    "length_penalty" : 1.1,

    "num_io_workers" : BASELINE_NUM_IO_WORKERS,
    "cache_answers" : BASELINE_CACHE_ANSWERS,

    "pre_compute_captions" : False,

    "use_reranker" : BASELINE_USE_RERANKER,
    "reranker_model" : BASELINE_RERANKER_MODEL,
    "reranker_batch_size" : BASELINE_RERANKER_BATCH_SIZE,

    "use_rewriter" : BASELINE_USE_REWRITER,
    "rewriter_model" : BASELINE_REWRITER_MODEL,
    "domain_hints" : ""
}

# params left out due to long import times - please see settings_heavy.py
# BASELINE_DEVICE, BASELINE_DTYPE, BASELINE_CAPTION_MODEL_ID
# BASELINE_CAP_PROCESSOR, BASELINE_CAP_MODEL      

# rag types
BENCHMARKER_SET_SIZE = 30 # slice size for benchmarking dataset

EVALUATION_PROMPT = """
Expected answer: {expected_answer}
Actual answer: {actual_answer}
---
Does the actual answer contain the expected answer? Respond only 'true' or 'false'.
"""

# questions sorted from roughly "easier" to more "difficult"
RAG_QUESTION_LIST: list[tuple[str, str]] = [
    # 1. numeric, quantitative information retrieval -> very specific questions found on particular passages
    (
        "How many meters tall is the 270 Park Avenue supertall skyscraper by Foster + Partners? (Answer with the number only)",
        "423"
    ),
    (
        "What stadium has a 100,000 seat capacity?  (Answer with the number only)",
        "Manchester United stadium"
    ),
    (
        "How many storeys does the Citadel headquarters supertall skyscraper in Miami have?  (Answer with the number only)",
        "54"   
    ),
    (
        "How many hotel rooms does the Miami skyscraper have? (Answer with the number only)",
        "212"
    ),
    (
        "How many residences will the Vela Viento skyscraper in Dubai house? (Answer with the number only)",
        "92"
    ),

    # 2. project-feature related, qualitative information
    (
        "Which project aims to achieve the WELL Community Gold rating? (Answer with the address only)",
        "18 Blackfriars Road"
    ),
    (
        "Which project has double-height atriums and landscaped terraces? (Answer with the name only)",
        "China Merchants Bank"  
    ),
    (
        "Which project creates a waterside community? (Answer with the name only)",
        "Xicen Science and Technology Centre"
    ),
    (
        "Which project has an earthy material palette? (Answer with the name only)",
        "Bodegas Faustino winery"
    ),
    (
        "Which project has a multi-storey glass courtyard? (Answer with the name only)",
        "The Edge"
    ),

    # 3. location-related, qualitative information
    (
        "In which city is the 18 Blackfriars Road development located? (Answer with the name of the city only)",
        "London"
    ),
    (
        "Where is the building with two separate glass volumes?",
        "Greenville"  
    ),
    (
        "Where is the Apple store TRX? (Answer with the name of the city only)",
        "Kuala Lumpur"
    ),
    (
        "Where in London is 50 Electric Boulevard located? (Answer with no more than five words)",
        "Battersea Power Station"
    ),
    (
        "Where are the Gadigal and Barangaroo stations? (Answer with name of the city only)",
        "Sydney"
    )
]

# SOURCE IDS AND PASSAGES - refer to services/rag/db/txt/chunks.txt
# a list of lists of relevant sources IDs - for some questions only 1 source is relevant
RAG_EXPECTED_SOURCES: list[list[str]] = [
    # 1. quantitative
    ["22_1"],
    ["1_1"],
    ["6_2"],
    ["6_1"],
    ["10_2"],
    # 2. qualitative
    ["35_1"],
    ["23_3"],
    ["13_1"],
    ["14_1"],
    ["15_1"],
    # 3. location
    ["35_1"],
    ["24_1"],
    ["25_1"],
    ["28_1"],
    ["31_1"]
]

# expected passages
RAG_EXPECTED_REFERENCES: list[list[str]] = [
    # 1. quantitative
    ["the 1,388-foot (423-metres) Foster + Partners-designed skyscraper is nearing completion on Park Avenue"],

    ["The 100,000-seat capacity"],

    ["Designed at 54 storeys"],

    ["212 hotel rooms"],

    ["92 residences"],

    # 2. qualitative
    ["18 Blackfriars Road development"],

    [""""double-height atriums and landscaped terraces to provide breakout space
     connected to the offices. The China Merchants Bank Headquarters"""],

    ["""Xicen Science and Technology Centre, a mixed-use development located within the 
     Yangtze River Delta region in Shanghai, China. Informed by the region's historic 
     water towns, the development by Foster + Partners will utilise the site's 
     new and existing waterways to create a "waterside community"""],

    ["""Bodegas Faustino winery in Spain, which features an earthy material palette"""],

    ["""multi-storey glass courtyard. Located in the upscale Carrasco neighbourhood of 
     Montevideo along the beachside avenue Rambla Tomas Berreta, The Edge"""],

    # 3. other qualitative
    ["18 Blackfriars Road development in London"],

    ["""Located in Greenville, the Greenville County Administration Building 
     contains offices and is part of a wider redevelopment of the county's central 
     square by RocaPoint Partners. British studio Foster + Partners has completed an 
     administrative building in South CarolinaThe building consists of two separate glass volumes"""],

    ["""store in Kuala Lumpur in an underground shop topped
     with a dome, designed by UK studio Foster + Partners. Named Apple The Exchange TRX"""],

    ["""50 Electric Boulevard, an 18,580-square-metre
     workplace within London's Battersea Power Station development"""],
     
    ["""Gadigal and Barangaroo stations, which form part of the first phase of the Sydney Metro City"""]

]