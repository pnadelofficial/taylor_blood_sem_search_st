#!/bin/sh
if [ -d '/taylor_blood_sem_search_st']; then 
    git pull
    echo "Running app..."
    streamlit run npi_st.py
else
    git clone https://github.com/pnadelofficial/taylor_blood_sem_search_st.git
    cd taylor_blood_sem_search_st
    pip install -r requirements.txt
    echo 'Running app...'
    streamlit run npi_st.py
fi