Make sure cuda version >= 11.8 is installed on host machine
Make sure docker and nvidia container toolkit is installed on host machine (https://docs.docker.com/config/containers/resource_constraints/#gpu)

`docker compose up`

Deploy all of the flows
`cd prefect-flows && python backfill_dataset.py`