# Knowledge Graph-Based Question Answering System

This repository contains the implementation of a Question Answering (QA) system that utilizes a knowledge graph constructed using the NetworkX framework. The system processes queries through a language model, identifying relevant information within the graph to generate responses.

## Project Overview

The Knowledge Graph-Based QA System aims to effectively answer questions by referencing a structured knowledge graph containing various entities and relations. This system explores the use of large language models (LLMs) in conjunction with graph-based data to enhance the accuracy and relevance of answers provided.

## Features

- **Knowledge Graph Construction**: Utilizes NetworkX to build a multi-graph representing movies, actors, directors, genres, and more.
- **Query Processing**: Implements advanced techniques to identify and aggregate relevant graph regions in response to user queries.
- **Integration with LLMs**: Leverages language models to interpret and answer questions based on the graph data.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/moeinsh78/KGQA-cs886.git
cd KGQA-cs886
pip install -r requirements.txt
