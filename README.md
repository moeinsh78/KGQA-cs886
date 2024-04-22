# Knowledge Graph-Based Question Answering System

This repository contains the implementation of a Question-Answering (QA) system that utilizes a knowledge graph to store information. The system processes queries through a large language model, asking it to answer the query based on the information retrieved from the knowledge graph. 

## Project Overview

The Knowledge Graph-Based QA system aims to answer questions by referring to a structured knowledge graph containing various entities and relations representing movies, actors, directors, genres, and more. The knowledge graph and the tests used present in the [MetaQA dataset](https://github.com/yuyuz/MetaQA) were used in this project. This system explores the use of large language models (LLMs) in conjunction with graph-based data to enhance the accuracy and relevance of the answers provided. The idea is to improve the baseline retrieval-augmented generation (RAG) technique by populating the LLM's context window with the most relevant information. The LLM used in this project is "gpt-3.5-turbo". 


![overview_image](assets/KGQA-arch.png)


## Installation

We prompted the system with samples of questions which answering them requires expanding a graph nodes for one, two, or three hops. During this process, the LLM response, expected results, and execution logs (showing the LLM prompt and response) in the "results" directory. The "results/x-hop-sample-output.txt" files contain the LLM response. Information stored in "results/x-hop-sample-log.txt" files contain the full LLM prompt (question + domain graph knowledge) and response, and the expected result for each line of LLM output is stored in its respective line in "results/x-hop-expected.txt" files. 

However, قunning this project involves sending requests to a gpt-3.5-turbo client to and requires valid OpenAI API key. Additionally, LLM responses are not always consistent, and diverge from providing the output alligned to the expected forms, which makes it difficult to parse and evaluate its performance. Hence, the above-mentioned files are provided to demonstrate model's performance, with the output file have been fixed in terms of formatting. 

Clone the repository and install the required packages:

```bash
git clone https://github.com/moeinsh78/KGQA-cs886.git
cd KGQA-cs886
