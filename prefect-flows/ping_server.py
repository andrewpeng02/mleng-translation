import os
import requests
import time
import mlflow
import shutil

from prefect import flow
from prefect.deployments import DeploymentImage


@flow(log_prints=True)
def main_flow():
    if "TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["TRACKING_URI"])

    assert requests.get("https://translate.andrewpeng.dev/").status_code == 200
    test_data = requests.post(
        "https://translate.andrewpeng.dev/api/predict", json={"input": "Hi"}
    ).json()
    assert "id" in test_data
    assert test_data["output"] == "Salut."

    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(
            "transformer-translation", "champion"
        )
    except:
        # retry
        time.sleep(5)
        version = client.get_model_version_by_alias(
            "transformer-translation", "champion"
        )

    # model should be updated in 1.5hr since last trained
    if time.time() - version.creation_timestamp / 1000 > 60*90:
        assert version.version == test_data["version"]
    print("All tests passed!")
    


if __name__ == "__main__":
    # main_flow()
    if os.path.isdir("shared"):
        shutil.rmtree("shared")
    shutil.copytree("../shared", "shared")
    main_flow.deploy(
        name="ping-server",
        work_pool_name="my-aci-pool-cpu",
        image=DeploymentImage(
            name="prefect-flows-image:latest",
            platform="linux/amd64",
            dockerfile="Dockerfile"
        ),
        interval=60*60 # run every hour
    )
