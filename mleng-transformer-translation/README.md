# transformer-translation
Using Pytorch's nn.Transformer module to create an english to french neural machine translation model. Details in https://andrewpeng.dev/transformer-pytorch/.

# Installation
To install the prerequisites into a conda environment, run
``` 
conda env create -f environment.yml
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```
# Set up
Start mlflow server: `mlflow server --host 127.0.0.1 --port 8000`
Start prefect server: `prefect server start --host 127.0.0.1 --port 4200` and `prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api`
`prefect work-pool create --type process my-pool`
`prefect worker start -p my-pool`
`prefect deploy orchestrate.py:main_flow -n deployment-v1 -p my-pool`
`prefect deployment run main-flow/deployment-v1`

# Manually running
## Training
Install and extract the english-french dataset from http://www.manythings.org/anki/ into the data/raw folder. Then run process_tatoeba_data.py, preprocess_data.py, then train_cli.py

## Inference
Run translate-sentence.py, which uses the transformer.pth model in /output. 
