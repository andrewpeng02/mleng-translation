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
- Make sure cuda version >= 11.8 is installed on host machine
- Make sure docker and nvidia container toolkit is installed on host machine (https://docs.docker.com/config/containers/resource_constraints/#gpu)
- To deploy, ... 

### Other flows
The code is located in prefect-flows, and consist of ...