import os
import shutil
import requests
import zipfile
from datetime import timedelta

import mlflow
from mlflow.entities import Run
from prefect import flow, task, runtime
from prefect.deployments import DeploymentImage
import great_expectations as gx
import pandas as pd
from dotenv import load_dotenv
load_dotenv("../.env")

from training_helpers import process_tatoeba_data
from training_helpers import preprocess_data
from training_helpers import train_cli
from shared import db_tables, db_helper


@task
def retrieve_dataset():
    db_helper.create_tables_if_not_exist()
    if db_helper.dataset_row_count() == 0:
        download_dataset()
        dataset = process_tatoeba_data_task()
        db_helper.bulk_insert_dataset(dataset)

    row_data = db_helper.get_model_execution_to_update()
    if len(row_data) > 0:
        # data validation
        validate_row_data(row_data)
        db_helper.bulk_insert_dataset(row_data)

    # retrieve dataset
    return db_helper.get_dataset()


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


def process_tatoeba_data_task():
    dataset_rows = process_tatoeba_data.main()
    shutil.rmtree("data/raw")
    return dataset_rows


def validate_row_data(row_data):
    df = pd.DataFrame(
        {
            "english": [row["english"] for row in row_data],
            "french": [row["french"] for row in row_data],
        }
    )
    context = gx.get_context()
    validator = context.sources.pandas_default.read_dataframe(df)
    validator.expect_column_values_to_not_be_null("english", mostly=0.8)
    validator.expect_column_values_to_not_be_null("french", mostly=0.8)
    validator.expect_column_value_lengths_to_be_between(
        "english", min_value=1, max_value=2000, mostly=0.8
    )
    validator.expect_column_value_lengths_to_be_between(
        "french", min_value=1, max_value=2000, mostly=0.8
    )
    validator.save_expectation_suite(discard_failed_expectations=False)

    context.add_or_update_checkpoint(name="checkpoint", validator=validator)

    run = context.run_checkpoint("checkpoint")
    if not run.success:
        raise Exception(f"Data validation failed: {run}")


@task
def preprocess_data_task(eng_dataset, fra_dataset):
    if os.path.isdir("data/processed"):
        shutil.rmtree("data/processed")
    os.makedirs("data/processed/en")
    os.makedirs("data/processed/fr")

    preprocess_data.main(eng_dataset, fra_dataset)


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

    run: Run = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.group_id = '{runtime.flow_run.id}'",
        order_by=["metrics.validation_loss ASC"],
        max_results=1,
    ).iloc[0]

    # promote model
    model_version = mlflow.register_model(
        f"runs:/{run.run_id}/model", "transformer-translation"
    )

    try:
        client = mlflow.MlflowClient()
        run_id = client.get_model_version_by_alias(
            "transformer-translation", "champion"
        ).run_id
        challenger_val_loss = run.data.metrics["validation_loss_final"]
        champion_val_loss = mlflow.get_run(run_id).data.metrics["validation_loss_final"]
        if (
            challenger_val_loss < champion_val_loss
            or abs(challenger_val_loss - challenger_val_loss) == 0.05
        ):
            client.set_registered_model_alias(
                "transformer-translation", "champion", model_version.version
            )
    except:
        print("No existing champion found. Setting current model as champion")
        client.set_registered_model_alias(
            "transformer-translation", "champion", model_version.version
        )


@flow(log_prints=True)
def main_flow():
    # mlflow.set_tracking_uri("http://0.0.0.0:8000")
    if "TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["TRACKING_URI"])
    eng_dataset, fra_dataset = retrieve_dataset()
    preprocess_data_task(eng_dataset, fra_dataset)
    train_and_optimize()
    promote_best_model()


if __name__ == "__main__":
    if os.path.isdir("shared"):
        shutil.rmtree("shared")
    shutil.copytree("../shared", "shared")
    # main_flow()
    # main_flow.deploy(
    #     name="train-model",
    #     work_pool_name="my-aci-pool",
    #     image=DeploymentImage(
    #         name="training-image:latest",
    #         platform="linux/amd64",
    #         dockerfile="Dockerfile",
    #     ),
    #     push=False
    # )
    main_flow.serve(name="train-process", interval=60*60*24*7)
