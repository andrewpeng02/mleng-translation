FROM ghcr.io/mlflow/mlflow:v2.9.2

RUN apt-get update && apt-get -y install curl
RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc \ 
    && curl https://packages.microsoft.com/config/debian/11/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \ 
    && apt-get update \ 
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \ 
    && echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc \
    && apt-get install -y unixodbc-dev

RUN pip install pyodbc azure-identity azure-storage-blob