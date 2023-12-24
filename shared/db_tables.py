import enum
from typing import Optional

from sqlalchemy import text, String, Enum, ForeignKey, Column
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER, DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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


class SourceEnum(enum.Enum):
    tatoeba: 1
    user: 2


class DatasetSource(Base):
    __tablename__ = "dataset_source"

    id: Mapped[int] = mapped_column(primary_key=True)
    id_other: Mapped[str]
    source = Column(Enum(SourceEnum))


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("dataset_source.id"))
    timestamp: Mapped[int]
    english: Mapped[str]
    french: Mapped[str]
