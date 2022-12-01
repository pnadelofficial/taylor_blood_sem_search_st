import streamlit as st
import pandas as pd
import spacy
from spacy.tokens import DocBin
import os
from stqdm import stqdm
from tqdm import tqdm
from semantic_search import SemanticSearch, sentence_tokenize
import gdown

@st.cache
def get_data():
    file_id = '1xdqk-wRVMmivpCPiTQs4ut5hYE9mvHth'
    output = 'serialized_data/spacy_model_output'
    gdown.download(
        f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}",
        output
    )
    npi1 = pd.read_csv('npi1.csv')
    npi2 = pd.read_csv('npi2.csv')
    npi3 = pd.read_csv('npi3.csv')
    npi4 = pd.read_csv('npi4.csv')
    base_df = pd.concat([npi1, npi2, npi3, npi4])
    npi_tokenized = sentence_tokenize(base_df).dropna()
    return npi_tokenized
npi_tokenized = get_data()

@st.cache(allow_output_mutation=True)
def load_model():
    return spacy.load('en_core_web_md')
nlp = load_model()

semantic_search = SemanticSearch(npi_tokenized, nlp)

entries = st.number_input('Choose number of excerpts.', min_value=1, value=5)
context_size = st.number_input('Choose context size (number of sentences before and after).', min_value=1, value=2)
cols_to_display = st.text_input('Enter names of columns to be displayed', 'title, date')
search_text = st.text_input('Search term', '')

search = semantic_search.search(
    'serialized_data/spacy_model_output', 
    search_text, 
    entries=entries, 
    context_size=context_size,
    streamlit=True,
    kwargs=cols_to_display
    )

st.markdown(
    f'<h2>{search[1]}</h2>'
    ,unsafe_allow_html=True
)

for i in range(len(search[0])):
    for col in search[0].columns[2:-1]:
        st.markdown(
            f'<small>{col.title()}: {search[0][col].to_list()[i]}</small>'
            ,unsafe_allow_html=True
        )
    st.markdown(
        f'<small>Similarity Score: {round(search[0].sent_docs.to_list()[i], 3)}</small>'
        ,unsafe_allow_html=True
    )
    st.markdown(
        f'<p>{search[0].context.to_list()[i]}</p>'
        ,unsafe_allow_html=True
    )
    st.markdown('<hr>', unsafe_allow_html=True)