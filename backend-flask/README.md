# backend-flask
This flask app serves as the backend to the translation model

# Installation
To install the prerequisites into a conda environment, run
``` 
conda env create -f environment.yml
python -m spacy download en_core_web_sm
```
# Set up
Dev: `python app.py`
Prod: `gunicorn --bind=127.0.0.1:9696 app:app`