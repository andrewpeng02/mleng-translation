FROM prefecthq/prefect:2.14.12-python3.11 as prefect

FROM anibali/pytorch:2.0.1-cuda11.8

COPY --from=prefect /opt/prefect /opt/prefect

RUN sudo apt-get update && sudo apt-get -y install curl
RUN curl https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc \ 
    && curl https://packages.microsoft.com/config/debian/11/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list \ 
    && sudo apt-get update \ 
    && sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18 \ 
    && echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc \
    && sudo apt-get install -y unixodbc-dev
    
RUN pip install prefect mlflow optuna spacy numpy tqdm einops azure-storage-blob azure-identity great_expectations pyodbc alibi-detect[pytorch]
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm

ENV TZ America/New_York

# Create the environment
COPY *.py /opt/prefect/mleng-transformer-translation/
COPY shared/*.py /opt/prefect/mleng-transformer-translation/shared/
COPY training_helpers/*.py /opt/prefect/mleng-transformer-translation/training_helpers/

WORKDIR /opt/prefect/mleng-transformer-translation/

USER root
RUN chown -R user:user /opt/prefect/mleng-transformer-translation
USER user