# mleng-translation
Live demo [here](https://translate.andrewpeng.dev)!

This is an English to French translation application powered by a deep learning model. It consists of the full stack application, the model, the training pipeline, and monitoring and observability:

![Architecture diagram](readme-files/Architecture%20Diagram.png)
*Architecture diagram*

- Full stack application
  - Users can access the website and recieve a translation in under 200ms, as well as give feedback on the translation
  - The underlying model automatically updates when a new version is available on the MLFlow server, training on new data
  - The technologies used includes React + tailwind for the frontend and Flask for the backend
  - Deployed on an Azure VM using Docker Compose, which uses a backend service, an MLFlow service, an nginx service, certbot for automatic SSL renewal, and Promethius + Grafana services
  - Grafana dashboard allows the admin to see average latency, endpoint usage, error reports, and more
  - Endpoint output is logged in a SQL table along with the model version, so that poor model outputs or errors can easily be traced back
- Model
  - The model is a sequence-to-sequence Transformer model trained in PyTorch
  - The model is also quantized and converted to JIT to reduce the latency, memory, and computational requirements
  - The data is pulled from a dataset SQL table, and most of the data originated from Tatoeba
- Training pipeline
  - The pipeline automatically trains and deploys the model to the backend without human intervention!
  - It fetches the newest version of the dataset from the SQL table, preprocesses it, trains the model, uploads the metrics and quantized model to MLFlow, and promotes it to production if the validation loss meets certain criteria
  - Pipeline was created using Prefect, and automatically runs every week
- Prefect flows
  - Another prefect flow includes `ping-server`, which ensures the server is up and runs smoke tests on the server every hour
  - If the server is down or a test doesn't pass, Prefect sends an email to me with the flow run information

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