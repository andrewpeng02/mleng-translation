import os
import urllib

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from shared.db_tables import Base, ModelExecution

params = urllib.parse.quote_plus(os.environ["AZURE_SQL_CONNECTIONSTRING"])
connection_string = "mssql+pyodbc:///?odbc_connect={}".format(params)
engine = create_engine(connection_string, echo=True)


def insert_into_model_execution(**kwargs):
    create_model_execution_table_if_not_exist()

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

def create_model_execution_table_if_not_exist():
    # Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    id = insert_into_model_execution(input="input", output="output")
    update_model_execution_output(id, "new output")
    # session = Session(engine)
    # blah = session.execute(select(ModelExecution))
    # print(blah.all())
    # session.close()
