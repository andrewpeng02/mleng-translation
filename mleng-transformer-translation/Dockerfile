FROM prefecthq/prefect:2.14.12-python3.11 as prefect

FROM anibali/pytorch:2.0.1-cuda11.8

COPY --from=prefect /opt/prefect /opt/prefect

RUN apt-get update && apt-get -y install curl
RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc \ 
    && curl https://packages.microsoft.com/config/debian/11/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \ 
    && apt-get update \ 
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \ 
    && echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc \
    && apt-get install -y unixodbc-dev
    
RUN pip install prefect mlflow optuna spacy numpy tqdm einops azure-storage-blob azure-identity great_expectations pyodbc
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm

ENV TZ America/New_York

# Create the environment
COPY *.py /opt/prefect/mleng-transformer-translation/

WORKDIR /opt/prefect/mleng-transformer-translation/
