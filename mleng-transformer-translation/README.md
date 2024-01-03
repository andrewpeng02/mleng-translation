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
Start prefect server: `prefect server start --host 127.0.0.1 --port 4200` 
Run the flow: `python orchestrate.py` (uncomment `main_flow()` and comment everything else in `main()`)