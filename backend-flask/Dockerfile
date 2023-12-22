FROM condaforge/miniforge3

WORKDIR /backend-flask

# Create the environment
COPY environment.yml .
RUN mamba env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["mamba", "run", "-n", "backend-flask", "/bin/bash", "-c"]

RUN python -m spacy download en_core_web_sm
RUN mkdir prometheus_multiproc_dir

# Copy application
COPY app.py gunicorn.conf.py .
ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "backend-flask", "gunicorn", "--bind=0.0.0.0:9696", "--workers=2", "--threads=2", "--timeout=120", "app:app"]