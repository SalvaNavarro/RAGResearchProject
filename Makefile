.PHONY: run db data
ENV=PE7046
ENV_CPU=PE7046_CPU

# 1. CONDA ENVIRONMENT (CUDA version)
env-create:
		mamba env create -n $(ENV) -f environment.yml

env-update:
		mamba env update -n $(ENV) -f environment.yml

env-remove:
		conda env remove -n $(ENV)

# CPU version if CUDA not available
# CPU environment stability has not been fully tested for all modules
env-create-cpu:
		mamba env create -n $(ENV_CPU) -f environment-cpu.yml

env-update-cpu:
		mamba env update -n $(ENV_CPU) -f environment-cpu.yml

env-remove-cpu:
		conda env remove -n $(ENV_CPU)

# 2. FIRST RUN SETUP
setup:
		python -m utility.scraper
		python -m utility.db_setup
		python -m services.rag --reset true

# 3. RUN FLASK APP
run:
		python run.py

# 4. INDEPENDENT MODULE RUNNER COMMANDS
# scrape command
scraper:
		python -m utility.scraper

# setup db (both app db and RAG db)
db:
		python -m utility.db_setup
		python -m services.rag --reset true
		python -m services.rag --reset true --mode benchmarking

# data analysis module
da:
		python -m utility.data_analysis

# RAG-specific commands
rag:
		python -m services.rag
		
eval:
		python -m utility.rag_evaluation

eval-bl:
		python -m utility.rag_evaluation --mode baseline

eval-bm:
		python -m utility.rag_evaluation --mode benchmarking
