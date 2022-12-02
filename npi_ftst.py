import pandas as pd
from numpy import dot
from numpy.linalg import norm
from dateutil.parser import parse
import streamlit as st
import fasttext
import fasttext.util
import nltk
from stqdm import stqdm
stqdm.pandas()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    #fasttext.util.download_model('en', if_exists='ignore')  # English
    st.file_uploader('Update FastText model.', type='bin')
    ft = fasttext.load_model('cc.en.300.bin')
    return ft
ft = load_model()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_data():
    npi1 = pd.read_csv('npi1.csv')
    npi2 = pd.read_csv('npi2.csv')
    npi3 = pd.read_csv('npi3.csv')
    npi4 = pd.read_csv('npi4.csv')
    base_df = pd.concat([npi1, npi2, npi3, npi4])

    sentences = base_df.copy()
    sentences['sents'] = sentences.text.apply(nltk.sent_tokenize)
    sentences = sentences.reset_index().drop('text',axis=1)
    sentences = sentences.rename(columns={'index':'org_index'})
    sentences = sentences.explode('sents').dropna().reset_index(drop=True)
    sentences['embedding'] = sentences.sents.progress_apply(lambda x: ft.get_sentence_vector(x))

    return sentences
sentences = get_data()

def cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def search(search_term, sentences, entries=5, context_size=4):
    if ' ' in search_term:
        search_emb = ft.get_sentence_vector(search_term)
    else:
        search_emb = ft.get_word_vector(search_term)

    sentences['sim_score'] = sentences['embedding'].progress_apply(lambda x: cosine_sim(search_emb, x))
    df = sentences.sort_values('sim_score', ascending=False)[0:entries].reset_index().rename(columns={'index':'org'})

    def get_context(org, context_size):
        context = sentences.iloc[org].sents
        for i in range(context_size):
            if (i < len(df)) and (i > 0):
                context = sentences.iloc[org-i].sents + '\n' + context
                context = context + '\n' + sentences.iloc[org+i].sents
        return context
    df['context'] = df['org'].apply(lambda x: get_context(x, context_size))
    
    return df, search_term

search_term = st.text_input('Search any term.', '')
entries = st.number_input('Choose number of excerpts.', min_value=1, value=5)
context_size = st.number_input('Choose size of excerpts (number of sentences before and after sentence of interest).', min_value=1, value=5)
if search_term != '':
    df, search_term = search(search_term, sentences, entries=entries, context_size=context_size)

    st.markdown(
        f'<h2>{search_term}</h2>'
        ,unsafe_allow_html=True
    )

    for i in range(entries):
        st.markdown(
            f'<small>Similarity Score: {round(df.sim_score.to_list()[i], 3)}</small>'
            ,unsafe_allow_html=True
        )
        st.markdown(
            f'<small>Title of document: {df.title.to_list()[i]}</small>'
            ,unsafe_allow_html=True
        )
        st.markdown(
            f'<small>Date posted: {df.date.to_list()[i]}</small>'
            ,unsafe_allow_html=True
        )
        st.markdown(
            f'<p>{df.context.to_list()[i]}</p>'
            ,unsafe_allow_html=True
        )
        st.markdown('<hr>', unsafe_allow_html=True)  


    