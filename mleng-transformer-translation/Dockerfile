FROM prefecthq/prefect:2.14.12-python3.11 as prefect

FROM anibali/pytorch:2.0.1-cuda11.8

COPY --from=prefect /opt/prefect /opt/prefect
RUN pip install prefect mlflow optuna spacy numpy tqdm einops azure-storage-blob azure-identity
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm

ENV TZ America/New_York

# Create the environment
COPY *.py /opt/prefect/mleng-transformer-translation/

WORKDIR /opt/prefect/mleng-transformer-translation/
