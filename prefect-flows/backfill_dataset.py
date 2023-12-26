import os
import shutil

from prefect import flow
from prefect.deployments import DeploymentImage

@flow(log_prints=True)            
def main_flow(name: str = "world"):           
    print(f"Hello {name}! I'm a flow running on an Azure Container Instance!")


if __name__ == "__main__":
    if os.path.isdir("shared"):
        shutil.rmtree("shared")
    shutil.copytree("../shared", "shared")
    main_flow.deploy(
        name="backfill-dataset", 
        work_pool_name="my-aci-pool-cpu",
        image=DeploymentImage(                                                 
            name="prefect-flows-image:latest",
            platform="linux/amd64",
            dockerfile="Dockerfile"
        )                                                                      
    )     