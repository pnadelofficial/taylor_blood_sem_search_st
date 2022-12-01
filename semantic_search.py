import pandas as pd
import spacy
from spacy.tokens import DocBin
import os
from tqdm import tqdm
from IPython.core.display import display, HTML
import nltk
import streamlit as st
from stqdm import stqdm
stqdm.pandas()

if not os.path.isdir('serialized_data'):
    os.mkdir('serialized_data')

def sentence_tokenize(df, col_name='text'):
    df['sents'] = df[col_name].apply(nltk.sent_tokenize)
    df_explode = df.explode('sents')
    return df_explode

class SemanticSearch():

    def __init__(self, df, nlp) -> None:
        self.df = df
        self.nlp = nlp

    def spacyify(self,col_name,f_name='serialized_data/spacy_model_output', streamlit=False):
        texts = self.df[col_name].to_list()
        doc_bin = DocBin()
        print('Reading texts...')
        if streamlit == True:
            progress_message = st.empty()
            progress_bar = st.progress(0)
            progress_status = st.empty()
            for i, doc in enumerate(self.nlp.pipe(texts)):
                doc_bin.add(doc)
                progress_message.write('Reading texts...')
                progress_status.write(f'{round((i/len(texts)*100), 1)}% complete')
                progress_bar.progress((i/len(texts))+(1/len(texts)))
        else:
            for doc in tqdm(self.nlp.pipe(texts), total=len(texts)):
                doc_bin.add(doc)
        print('Done.')
        bytes_data = doc_bin.to_bytes()

        f = open(f'{f_name}','wb')
        f.write(bytes_data)
        f.close()

    def search(self, doc_bin_path, search_text, entries=5, context_size=2, streamlit=False, **kwargs):
        if ' ' in search_text:
            search_vec = self.nlp(search_text)
        else:
            search_vec = self.nlp.vocab[search_text]
        
        bytes_file = open(doc_bin_path,'rb').read()
        doc_bin = DocBin().from_bytes(bytes_file)
        docs = pd.Series(doc_bin.get_docs(self.nlp.vocab))
        sim = pd.DataFrame({'sents':self.df.sents.to_list(), 'sent_docs':docs.to_list()})

        if streamlit == True:
            sim_score = sim['sent_docs'].progress_apply(lambda x: x.similarity(search_vec)).sort_values(ascending=False)[0:entries]
        else:
            sim_score = sim['sent_docs'].apply(lambda x: x.similarity(search_vec)).sort_values(ascending=False)[0:entries]
        sim_df = sim_score.reset_index().rename(columns={'index':'org_idx'})
        
        if streamlit == True:
            for i, col in enumerate(list(kwargs.values())[0].split(',')):
                sim_df[col] = sim_df['org_idx'].apply(lambda x: self.df[col.strip()].iloc[x])
        else:
            for key,value in kwargs.items():
                sim_df[key] = sim_df['org_idx'].apply(lambda x: self.df[value].iloc[x])

        def create_context(org, context_size):
            context = list(sim.sent_docs.iloc[org].sents)[0].text
            for i in range(context_size):
                if (i < len(sim)) and (i > 0):
                    context = list(sim.sent_docs.iloc[org-i].sents)[0].text + '\n' + context
                    context = context + '\n' + list(sim.sent_docs.iloc[org+i].sents)[0].text
            return context
        sim_df['context'] = sim_df['org_idx'].apply(lambda x: create_context(x, context_size))

        return sim_df, search_text

    def displaySearch(self, search_df, search_text):
        display(HTML(f'<h2>{search_text}</h2>'))
        display(HTML('<br>'))
        for i in range(len(search_df)):
            for col in search_df.columns[2:-1]:
                display(HTML(f'<small><i>{col.title()}: {search_df[col].to_list()[i]}</i></small>'))
            display(HTML(f'<small>Similarity Score: {round(search_df.sent_docs.to_list()[i], 3)}</small>'))
            display(HTML(f'<p>{search_df.context.to_list()[i]}</p>'))
            display(HTML('<br>'))

    def searchWordOrPhrase(self, doc_bin_path,entries=5, context_size=2, **kwargs):
        search_term = input('Enter search term:')
        search = self.search(doc_bin_path,search_term,entries=entries, context_size=context_size, **kwargs)
        self.displaySearch(search[0],search[1])