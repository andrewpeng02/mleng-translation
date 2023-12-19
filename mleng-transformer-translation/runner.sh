#!/bin/sh
nohup mamba run -n mleng-transformer-translation \
      prefect server start --host 0.0.0.0 --port 4200 > logs/prefect_server.out &
sleep 3
nohup mamba run -n mleng-transformer-translation \
      prefect worker start -p my-pool > logs/worker.out &
mamba run -n mleng-transformer-translation \
      mlflow server --host 0.0.0.0 --port 8000