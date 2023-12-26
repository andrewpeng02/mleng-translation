from prefect import flow
from prefect.deployments import DeploymentImage

import torch

@flow(log_prints=True)            
def my_flow(name: str = "world"):           
    print(torch.cuda.is_available())               
    print(f"Hello {name}! I'm a flow running on an Azure Container Instance!")


if __name__ == "__main__":
    my_flow.deploy(
        name="train-model", 
        work_pool_name="my-aci-pool",
        image=DeploymentImage(                                                 
            name="training-image:latest",
            platform="linux/amd64",
            dockerfile="Dockerfile"
        )                                                                      
    )       
