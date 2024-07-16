from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pandas import DataFrame as PandasDataFrame

from splink.internals.input_column import InputColumn
from splink.internals.splink_dataframe import SplinkDataFrame

from sqlglot.dialects import Dialect

logger = logging.getLogger(__name__)

Dialect["snowflake"]
if TYPE_CHECKING:
    from .database_api import SnowflakeAPI


class SnowflakeDataFrame(SplinkDataFrame):
    db_api: SnowflakeAPI

    @property
    def columns(self) -> list[InputColumn]:
        sql = f"select * from {self.physical_name} limit 1"
        snowflake_df = self.db_api._execute_sql_against_backend(sql)

        col_strings = list(snowflake_df.columns)
        return [InputColumn(c, sql_dialect="snowflake") for c in col_strings]

    def validate(self):
        pass

    def as_record_dict(self, limit=None):
        sql = f"select * from {self.physical_name}"
        if limit:
            sql += f" limit {limit}"

        data = self.as_pandas_dataframe(limit=limit)
        data.columns = data.columns.str.lower()
        return data.to_dict(orient="records")

    def _drop_table_from_database(self, force_non_splink_table=False):
        if self.db_api.break_lineage_method == "delta_lake_table":
            self._check_drop_table_created_by_splink(force_non_splink_table)
            self.db_api.delete_table_from_database(self.physical_name)
        else:
            pass

    def as_pandas_dataframe(self, limit: int = None) -> PandasDataFrame:
        sql = f"select * from {self.physical_name}"
        if limit:
            sql += f" limit {limit}"

        return self.db_api._execute_sql_against_backend(sql).toPandas()

    def as_snowflake_dataframe(self):
        return self.db_api.snowflake.table(self.physical_name)

    def to_table(self, table_name, overwrite=False):
        snowflake_df = self.as_snowflake_dataframe()
        snowflake_df.write.mode("overwrite" if overwrite else "error").saveAsTable(table_name)
