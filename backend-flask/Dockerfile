FROM condaforge/miniforge3

RUN apt-get update && apt-get -y install curl
RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc \ 
    && curl https://packages.microsoft.com/config/debian/11/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \ 
    && apt-get update \ 
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \ 
    && echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc \
    && apt-get install -y unixodbc-dev

WORKDIR /backend-flask

# Create the environment
COPY environment.yml .
RUN mamba env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["mamba", "run", "-n", "backend-flask", "/bin/bash", "-c"]

RUN python -m spacy download en_core_web_sm
RUN mkdir prometheus_multiproc_dir

# Copy application
COPY app.py db_helper.py gunicorn.conf.py .
ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "backend-flask", "gunicorn", "--bind=0.0.0.0:9696", "--workers=1", "--threads=1", "--timeout=120", "app:app"]