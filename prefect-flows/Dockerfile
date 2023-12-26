FROM prefecthq/prefect:2.14-python3.11

RUN apt-get update && apt-get -y install curl
RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc \ 
    && curl https://packages.microsoft.com/config/debian/11/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \ 
    && apt-get update \ 
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \ 
    && echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc \
    && apt-get install -y unixodbc-dev

# Create the environment
COPY requirements.txt .
RUN pip install -r requirements.txt 

# Copy application
COPY *.py .
COPY shared shared