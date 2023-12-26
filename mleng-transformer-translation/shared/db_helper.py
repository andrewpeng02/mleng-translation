import os
import urllib

from sqlalchemy import create_engine, select, insert, func
from sqlalchemy.orm import Session

from shared.db_tables import Base, ModelExecution, Dataset, DatasetSource, SourceEnum

params = urllib.parse.quote_plus(os.environ["AZURE_SQL_CONNECTIONSTRING"])
connection_string = "mssql+pyodbc:///?odbc_connect={}".format(params)
engine = create_engine(connection_string)  # , echo=True)


def insert_into_model_execution(**kwargs):
    create_tables_if_not_exist()

    session = Session(engine)
    model_execution = ModelExecution(**kwargs)
    session.add(model_execution)

    session.flush()
    session.refresh(model_execution)
    id = model_execution.id
    session.commit()
    session.close()

    return id


def update_model_execution_output(id, output):
    session = Session(engine)

    model_execution = session.get(ModelExecution, id)
    model_execution.output = output

    session.commit()
    session.close()


def update_model_execution_user_label(id, user_label):
    session = Session(engine)

    model_execution = session.get(ModelExecution, id)
    model_execution.user_label = user_label

    session.commit()
    session.close()


def create_tables_if_not_exist():
    # Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def dataset_row_count():
    session = Session(engine)

    count = session.execute(select(func.count(Dataset.id))).scalar()
    session.close()
    return count if count is not None else 0


def bulk_insert_dataset(dataset_rows):
    """Bulk inserts a list of rows into the Dataset table

    Args:
        dataset_rows (List[Dict[english: str, french: str, source: SourceEnum, id_other: string]]): List of dictionaries
    """
    if len(dataset_rows) == 0:
        return
    bulk_dataset_sources = [
        DatasetSource(id_other=row["id_other"], source=row["source"])
        for row in dataset_rows
    ]
    session = Session(engine)
    for i in range(int((len(bulk_dataset_sources) + 9999) / 10000)):
        print(f"{i} / {int((len(bulk_dataset_sources) + 9999) / 10000)}")
        batch_dataset_sources = bulk_dataset_sources[
            i * 10000 : min((i + 1) * 10000, len(bulk_dataset_sources))
        ]
        batch_dataset_rows = dataset_rows[
            i * 10000 : min((i + 1) * 10000, len(dataset_rows))
        ]

        session.bulk_save_objects(batch_dataset_sources, return_defaults=True)
        bulk_dataset_rows = [
            Dataset(source_id=source.id, english=row["english"], french=row["french"])
            for source, row in zip(batch_dataset_sources, batch_dataset_rows)
        ]
        session.bulk_save_objects(bulk_dataset_rows)
        session.commit()
    session.close()


def get_model_execution_to_update():
    session = Session(engine)
    most_recent_timestamp = session.execute(
        select(func.max(Dataset.timestamp))
    ).scalar()
    if most_recent_timestamp is None:
        return

    valid_rows = session.execute(
        select(ModelExecution.id, ModelExecution.input, ModelExecution.label)
        .where(ModelExecution.timestamp > most_recent_timestamp)
        .where(ModelExecution.label != None)
    )
    row_data = [
        {
            "english": row.input,
            "french": row.label,
            "source": SourceEnum.user,
            "id_other": row.id,
        }
        for row in valid_rows
    ]

    session.close()
    return row_data


def get_dataset():
    session = Session(engine)
    rows = session.execute(select(Dataset.english, Dataset.french))
    eng_dataset = []
    fra_dataset = []
    for row in rows:
        eng_dataset.append(row.english)
        fra_dataset.append(row.french)
    session.close()
    return eng_dataset, fra_dataset


if __name__ == "__main__":
    # id = insert_into_model_execution(input="input", output="output")
    # update_model_execution_output(id, "new output")
    create_tables_if_not_exist()
    # print(dataset_row_count())
    # session = Session(engine)
    # blah = session.execute(select(ModelExecution))
    # print(blah.all())
    # session.close()
