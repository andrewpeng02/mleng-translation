import os
import shutil
import time
import requests
import zipfile
from datetime import timedelta

import torch
import mlflow
from mlflow.entities import RunInfo
from prefect import flow, task, runtime

import process_tatoeba_data
import preprocess_data
import train_cli


@task(
    retries=3,
    retry_delay_seconds=1,
)
def download_dataset():
    """
    Downloads the fra-eng dataset from many-things

    Data originated from Tatoeba, a site with crowd sourced translations
    """

    if os.path.isdir("data/raw"):
        shutil.rmtree("data/raw")
    os.makedirs("data/raw")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
    }

    r = requests.get(
        "https://www.manythings.org/anki/fra-eng.zip",
        allow_redirects=True,
        headers=headers,
        timeout=10,
    )
    open("data/raw/fra-eng.zip", "wb").write(r.content)

    with zipfile.ZipFile("data/raw/fra-eng.zip", "r") as zip_ref:
        zip_ref.extractall("data/raw/fra-eng")
    os.rename("data/raw/fra-eng/fra.txt", "data/raw/fra.txt")
    os.remove("data/raw/fra-eng.zip")
    shutil.rmtree("data/raw/fra-eng")


@task
def process_tatoeba_data_task():
    process_tatoeba_data.main()


@task
def preprocess_data_task():
    if os.path.isdir("data/processed"):
        shutil.rmtree("data/processed")
    os.makedirs("data/processed/en")
    os.makedirs("data/processed/fr")

    preprocess_data.main()


@task
def train_and_optimize():
    train_cli.main(
        ["--auto_optimize", "True", "--group_id", runtime.flow_run.id],
        standalone_mode=False,
    )


@task
def promote_best_model():
    experiment = mlflow.search_experiments(
        filter_string="attribute.name = 'English to French Translation'"
    )[0]

    run: RunInfo = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.group_id = '{runtime.flow_run.id}'",  
        # filter_string=f"tags.group_id = 'c76d34f4-cff9-48c5-ba4e-99a91b106cf3'",  # 3b1355ef-a7a9-4408-a8f7-72b79c002be5
        order_by=["metrics.validation_loss ASC"],
        max_results=1,
    ).iloc[0]

    # promote model
    model_version = mlflow.register_model(
        f"runs:/{run.run_id}/model", "transformer-translation"
    )
    mlflow.MlflowClient().set_registered_model_alias(
        "transformer-translation", "champion", model_version.version
    )


@flow(log_prints=True)
def main_flow():
    mlflow.set_tracking_uri("http://0.0.0.0:8000")
    download_dataset()
    process_tatoeba_data_task()
    preprocess_data_task()
    train_and_optimize()
    promote_best_model()


if __name__ == "__main__":
    main_flow()
