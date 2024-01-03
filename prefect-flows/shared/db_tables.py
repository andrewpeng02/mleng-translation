import enum
from typing import Optional

from sqlalchemy import text, String, Enum, ForeignKey, Column
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER, DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ModelExecution(Base):
    __tablename__ = "andrew_model_execution"

    id = Column(UNIQUEIDENTIFIER, primary_key=True, server_default=text("newid()"))
    timestamp = Column(DATETIME, server_default=text("CURRENT_TIMESTAMP"))
    input: Mapped[str] = mapped_column(String(5000))
    output: Mapped[Optional[str]] = mapped_column(String(5000))
    error: Mapped[Optional[str]]
    label: Mapped[Optional[str]] = mapped_column(String(5000))
    user_label: Mapped[Optional[str]] = mapped_column(String(5000))

    def __repr__(self) -> str:
        return f"ModelExecution(id={self.id}, timestamp={self.timestamp}, input={self.input}, output={self.output}, label={self.label}, user_label={self.user_label})"


class SourceEnum(enum.Enum):
    tatoeba = 1
    user = 2


class DatasetSource(Base):
    __tablename__ = "andrew_dataset_source"

    id: Mapped[int] = Column(
        UNIQUEIDENTIFIER, primary_key=True, server_default=text("newid()")
    )
    id_other: Mapped[str]
    source = Column(Enum(SourceEnum))

    def __repr__(self) -> str:
        return f"DatasetSource(id={self.id}, id_other={self.id_other}, source={self.source})"


class Dataset(Base):
    __tablename__ = "andrew_dataset"

    id: Mapped[int] = Column(
        UNIQUEIDENTIFIER, primary_key=True, server_default=text("newid()")
    )
    source_id: Mapped[int] = mapped_column(ForeignKey("andrew_dataset_source.id", ondelete="CASCADE"))
    timestamp = Column(DATETIME, server_default=text("CURRENT_TIMESTAMP"))
    english: Mapped[str]
    french: Mapped[str]

    def __repr__(self) -> str:
        return f"Dataset(id={self.id}, source_id={self.source_id}, timestamp={self.timestamp}, english={self.english}, french={self.french})"
