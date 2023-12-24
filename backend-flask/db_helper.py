import enum
import os
import struct
from typing import Optional
import urllib

from sqlalchemy import create_engine, text, String, Enum, ForeignKey, Column, select
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER, DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from azure import identity

params = urllib.parse.quote_plus(os.environ["AZURE_SQL_CONNECTIONSTRING"])
connection_string = "mssql+pyodbc:///?odbc_connect={}".format(params)
engine = create_engine(connection_string, echo=True)


class Base(DeclarativeBase):
    pass


class ModelExecution(Base):
    __tablename__ = "model_execution"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, server_default=text("newid()"))
    timestamp = Column(DATETIME, server_default=text("CURRENT_TIMESTAMP"))
    input: Mapped[str] = mapped_column(String(2000))
    output: Mapped[Optional[str]] = mapped_column(String(2000))
    error: Mapped[Optional[str]]
    label: Mapped[Optional[str]] = mapped_column(String(2000))
    user_label: Mapped[Optional[str]] = mapped_column(String(2000))

    def __repr__(self) -> str:
        return f"ModelExecution(id={self.id}, timestamp={self.timestamp}, input={self.input}, output={self.output}, label={self.label}, user_label={self.user_label})"


# class SourceEnum(enum.Enum):
#     tatoeba: 1
#     user: 2

# class DatasetSource(Base):
#     __tablename__ = "dataset_source"

#     id: Mapped[int] = mapped_column(primary_key=True)
#     id_other: Mapped[int]
#     source: Mapped[Enum(SourceEnum)]

# class Dataset(Base):
#     __tablename__ = "dataset"

#     id: Mapped[int] = mapped_column(primary_key=True)
#     source_id: mapped_column(ForeignKey(dataset_source.id))
#     timestamp: Mapped[int]
#     english: Mapped[str]
#     french: Mapped[str]


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
