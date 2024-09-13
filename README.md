# :spaghetti: SG Personal Food Assistant :spaghetti:
##### Automatic Speech Recognition(ASR) with WhisperAI fine-tuned to localities in Singapore and using Llama 3.1 as a chatbot

# Table of Contents

1. [Background](#background)
2. [Purpose](#purpose)
3. [Methodology](#methodology)
4. [Data Dictionary](#data-dictionary)
5. [Models Used](#models-used)
6. [Summary of Findings](#summary-of-findings)
7. [Installation](#installation)
    - [Hardware](#hardware)
    - [Dependencies](#dependencies)
    - [File Format](#file-format)
    
## Background
Apart from our world class tourist attractions such as Gardens by the Bay and Marina Bay Sands,
one of the top three reasons which constantly appear on lists on why people should visit Singapore
is our Food and Hawker Culture. With tourism bringing in about SGD66 Billion in revenue
annually, of which almost SGD10 Billion of it are from food and beverages (2023, SingStat), this
SG Personal Food Assistant app will be able to assist in the planning of tourists’ trips to
Singapore, recommendation of local foods and also exploration of lesser known food havens in
Singapore, especially in one of our many hawker centres.

## Purpose
In this project, we will fine-tune WhisperAI to recognize locations in Singapore and create a personal assistant chatbot using Llama 3.1 to give recommendations on everything food-related in Singapore. This model can be further fine-tuned and kept up-to-date with RAG, to ensure that the food and restaurant recommendations given are from the latest sources. With this personal food assistant, we aim to promote food tourism and revenue to Singapore which may subsequently preserve and maybe even boost the dying local hawker culture in Singapore.

## Methodology
For the fine-tuning of WhisperAI, a script was created and volunteers were sought to record an audio file of them reading the script. The audio files were then manually split, labelled and augmented to produce about 2.5h of training data.

For the actual personal assistant chatbot, a Groq client was used to reduce the latency of the queries, and the query was fed to a Llama 3.1 8B Instant model. This resulted in near instant query retrieval between the chatbot and the server.

## Models Used
1. WhisperAI-small.en
2. Llama 3.1 8B Instant

## Summary of Findings

From the fine-tuning of the WhisperAI model, we managed to achieve a Word Error Rate (WER) of only 6.2%, with the addition of over 40 local words (locations in Singapore).

Albeit so, much more can be done to build a more accurate, robust and future-proof models for this use case. Future improvements that can be considered include:

1. Getting more training data for fine-tuning of WhisperAI
2. Fine-tuning WhisperAI to recognize local food names and entities
3. Fine-tuning Llama 3.1 with RAG to ensure up-to-date information on food and resturants in Singapore

## Installation

##### *Hardware*

An Intel i5 computer with 16GB RAM and an NVIDIA RTX4070 Super GPU was used to fine-tune the deep learning models in this project (~10min to run the full notebook)

##### *Dependencies*

Install the dependencies by opening your terminal in the project folder and run:

`pip install -r requirements.txt`

After installation, you should then be able to run the code smoothly :smile:

##### *File Format*

The WhisperAI fine tuning file is in .ipynb format and was created in VS Code and Jupyter Notebook :book:</br>
The `main.py` file is made with Streamlit to be run locally, however you may need to obtain a Groq API key to be able to run the Streamlit file, which sends queries to Llama 3.1 through the Groq client.
