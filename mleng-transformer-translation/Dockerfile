FROM prefecthq/prefect:2.14.12-python3.11 as prefect

FROM python/3.11.7-alpine3.18

COPY --from=prefect / /

# Create the environment
COPY *.py /opt/prefect/mleng-transformer-translation/

WORKDIR /opt/prefect/mleng-transformer-translation/
