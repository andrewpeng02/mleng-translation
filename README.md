## How to deploy to production
This project requires deploying the app and Prefect flows separately.

### Deploying the app
To deploy the app, ensure Docker and Docker Compose are installed. Clone the repository and create a `.env` file at the root of the repository. It should contain
- `BACKEND_URI` and `ARTIFACT_URI` passed to mlflow
- `AZURE_STORAGE_ACCESS_KEY` and `AZURE_STORAGE_CONNECTION_STRING` also passed to mlflow, optional if not using azure
- `AZURE_SQL_CONNECTIONSTRING` passed to the app and prefect flows to keep track of the dataset and logging, can pass any connection string 

Set up ssl by running `sudo ./init-letsencrypt.sh`, ensuring you have the correct domain name in `init-letsencrypt.sh` and `nginx/nginx.conf`.
Then, run `docker compose down && docker compose up -d` and the app should be available at port 443, grafana at port 3000, and mlflow at port 8000

### Deploying the Prefect flows
The Prefect flows are separated into two groups based on if it requires a GPU or not. 

### Training flow
The code is located in mleng-transformer-translation, and requires:
- GPU with >16gb of ram (alternatively, reduce num_tokens)
- To deploy, follow these steps

Set up once:
1. Log in to Prefect Cloud
2. `cd mleng-transformer-translation`
3. `conda env create -f environment.yml`

To deploy as much as you want:
1. `conda activate mleng-transformer-translation`
2. `cd mleng-transformer-translation`
3. `python3 orchestrate.py` with `AZURE_SQL_CONNECTIONSTRING` as an environment variable
4. Go into Prefect Cloud and run your deployment!

### Other flows
The code is located in prefect-flows, and consist of `ping_server.py`, which ensures server uptime.
Set up once: 
1. Log in to Prefect Cloud and provision a serverless push work pool (https://docs.prefect.io/latest/guides/deployment/push-work-pools/)
2. Install Azure CLI, `az login`, and `az acr login --name [NAME OF CONTAINER REGISTRY]`
3. `cd prefect-flows`
3. `pip install -r requirements.txt`
4. `prefect config set PREFECT_DEFAULT_DOCKER_BUILD_NAMESPACE=<docker-registry-url>/<organization-or-username>`

To deploy as much as you want:
1. `cd prefect-flows`
2. `python3 ping_server.py`