#!/bin/bash
python download_models.py
streamlit run app.py --server.port=8501 --server.enableCORS=false
