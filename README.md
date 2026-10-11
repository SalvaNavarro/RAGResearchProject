# Introduction

A Python solution, part of the PE7046 Research Project module at Northumbria University (NU),
built to research the application of RAG systems for knowledge management at
large architecture practices. A Makefile is included with the main commands
to set up and run the solution.

# Prerequisites

The project Python's environment uses conda manager, and mamba is recommended for faster installation.
Makefile uses mamba commands, which can be manually switched to conda commands if mamba is not available.

The system requires the software Ollama to be installed and running before the next steps
can proceed. This can be downloaded from https://ollama.com/download

# Preparing the environment

Please use command env-create to create the conda environment. This will install the
packages specified on the following files:

1. environment.yml -> holds the specification for all the main conda packages
2. requirements.txt -> will be called automatically from environment.yml

# How to use

On a first run, modules should be executed in this order, for all databases to get populated:

1. utility.scraper
2. utility.db_setup
3. utility.data_analysis
4. services.rag (reset==true)
5. app

Please refer to Makefile commands "setup", then "run" for a first run of the project.

The services.rag module will run inside the app module.
The services.rag_evaluation module can be run separately.
A test suite that checks essential functionality of the rag module is included inside the tests folder.

# Acknowledgements

Author: Salvador Cristobal Navarro Perez
Supervisor: Keerthana Jaganathan, NU
Second marker :Abbas Mehrabidavoodabadi, NU
Module tutor: Dr Ian Watson, NU
