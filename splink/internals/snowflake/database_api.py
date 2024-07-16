import logging
import math
import os
import re

import pandas as pd
import sqlglot
from numpy import nan

from snowflake.snowpark import DataFrame as snowpark_df
from pyspark.sql.utils import AnalysisException

from splink.internals.database_api import AcceptableInputTableType, DatabaseAPI

from splink.internals.dialects import SnowflakeDialect

from splink.internals.misc import (
    major_minor_version_greater_equal_than,
)

from .dataframe import SnowflakeDataFrame

logger = logging.getLogger(__name__)


class SnowflakeAPI(DatabaseAPI[snowpark_df]):
    sql_dialect = SnowflakeDialect()

    def __init__(
        self,
        *,
        session,
        break_lineage_method=None,
        catalog=None,
        database=None,
        # TODO: what to do about repartitions:
        repartition_after_blocking=False,
        num_partitions_on_repartition=None,
        register_udfs_automatically=True,
    ):
        super().__init__()
        # TODO: revise logic as necessary!
        self.break_lineage_method = break_lineage_method

        # these properties will be needed whenever spark is _actually_ set up
        self.repartition_after_blocking = repartition_after_blocking

        # TODO: hmmm breaking this flow. Lazy spark ??

        self.snowflake = session

        # if num_partitions_on_repartition:
        #     self.num_partitions_on_repartition = num_partitions_on_repartition
        # else:
        #     self.set_default_num_partitions_on_repartition_if_missing()

        self._set_splink_datastore(catalog, database)

        # TODO: also need to think about where these live:
        # self._drop_splink_cached_tables()
        # self._check_ansi_enabled_if_converting_dates()

        # TODO: also need to think about how to lineage in case snowflake
        # self._set_default_break_lineage_method()

    def _table_registration(
        self, input: AcceptableInputTableType, table_name: str
    ) -> None:
        if isinstance(input, dict):
            input = pd.DataFrame(input)
        elif isinstance(input, list):
            input = pd.DataFrame.from_records(input)

        if isinstance(input, pd.DataFrame):
            input = self._clean_pandas_df(input)
            input = self.snowflake.createDataFrame(input)

        input.createOrReplaceTempView(table_name)

    def table_to_splink_dataframe(
        self, templated_name: str, physical_name: str
    ) -> SnowflakeDataFrame:
        return SnowflakeDataFrame(templated_name, physical_name, self)

    def table_exists_in_database(self, table_name):
        query_result = self._execute_sql_against_backend(
            f"""show tables like '{table_name}' in database {self.splink_data_store}"""
        ).collect()
        if len(query_result) == 1:
            return True
        elif len(query_result) == 0:
            return False

    def _setup_for_execute_sql(self, sql: str, physical_name: str) -> str:
        sql = sqlglot.transpile(sql, read="snowflake", write="snowflake", pretty=True)[0]
        return sql

    def _cleanup_for_execute_sql(
        self, table: snowpark_df, templated_name: str, physical_name: str
    ) -> SnowflakeDataFrame:
        snowflake_df = self._break_lineage_and_repartition(
            table, templated_name, physical_name
        )

        # After blocking, want to repartition
        # if templated
        snowflake_df.write.save_as_table(physical_name, table_type="temporary", mode="overwrite")

        output_df = self.table_to_splink_dataframe(templated_name, physical_name)
        return output_df

    def _execute_sql_against_backend(self, final_sql: str) -> snowpark_df:
        return self.snowflake.sql(final_sql)

    def delete_table_from_database(self, name):
        self._execute_sql_against_backend(f"drop table {name}")

    @property
    def accepted_df_dtypes(self):
        return [pd.DataFrame, snowpark_df]

    def _clean_pandas_df(self, df):
        return df.fillna(nan).replace([nan, pd.NA], [None, None])

    def _set_splink_datastore(self, catalog, database):
        # snowflake.catalog.currentCatalog() is not available in versions of snowflake before
        # 3.4.0. In snowflake versions less that 3.4.0 we will require explicit catalog
        # setting, but will revert to default in snowflake versions greater than 3.4.0
        threshold = "3.4.0"

        # if (
        #     major_minor_version_greater_equal_than(self.snowflake.version, threshold)
        #     and not catalog
        # ):
        #     # set the catalog and database of where to write output tables
        #     catalog = (
        #         catalog if catalog is not None else self.snowflake.catalog.currentCatalog()
        #     )
        database = (
            database if database is not None else self.snowflake.get_current_database()
        )

        # this defines the catalog.database location where splink's data outputs will
        # be stored. The filter will remove none, so if catalog is not provided and
        # snowflake version is < 3.3.0 we will use the default catalog.
        self.splink_data_store = ".".join(
            [x for x in [database] if x is not None]
        )

    def _get_checkpoint_dir_path(self, snowflake_df):
        # https://github.com/apache/snowflake/blob/301a13963808d1ad44be5cacf0a20f65b853d5a2/python/pysnowflake/context.py#L1323 # noqa E501
        # getCheckpointDir method exists only in snowflake 3.1+, use implementation
        # from above link
        if not self.snowflake._jsc.sc().getCheckpointDir().isEmpty():
            return self.snowflake._jsc.sc().getCheckpointDir().get()
        else:
            # Raise checkpointing error
            snowflake_df.limit(1).checkpoint()

    def set_default_num_partitions_on_repartition_if_missing(self):
        parallelism_value = 200
        try:
            parallelism_value = self.snowflake.conf.get("snowflake.default.parallelism")
            parallelism_value = int(parallelism_value)
        except Exception:
            pass

        # Prefer snowflake.sql.shuffle.partitions if set
        try:
            parallelism_value = self.snowflake.conf.get("snowflake.sql.shuffle.partitions")
            parallelism_value = int(parallelism_value)
        except Exception:
            pass

        self.num_partitions_on_repartition = math.ceil(parallelism_value / 2)

    # TODO: this repartition jazz knows too much about the linker
    def _repartition_if_needed(self, snowflake_df, templated_name):
        # Repartitioning has two effects:
        # 1. When we persist out results to disk, it results in a predictable
        #    number of output files.  Some splink operations result in a very large
        #    number of output files, so this reduces the number of files and therefore
        #    avoids slow reads and writes
        # 2. When we repartition, it results in a more evenly distributed workload
        #    across the cluster, which is useful for large datasets.

        names_to_repartition = [
            r"__splink__df_comparison_vectors",
            r"__splink__df_blocked",
            r"__splink__df_neighbours",
            r"__splink__df_representatives",
            r"__splink__df_concat_with_tf_sample",
            r"__splink__df_concat_with_tf",
            r"__splink__df_predict",
        ]

        num_partitions = self.num_partitions_on_repartition

        # TODO: why regex not == ?
        if re.fullmatch(r"__splink__df_predict", templated_name):
            num_partitions = math.ceil(self.num_partitions_on_repartition)

        if re.fullmatch(r"__splink__df_representatives", templated_name):
            num_partitions = math.ceil(self.num_partitions_on_repartition / 6)

        if re.fullmatch(r"__splink__df_neighbours", templated_name):
            num_partitions = math.ceil(self.num_partitions_on_repartition / 4)

        if re.fullmatch(r"__splink__df_concat_with_tf_sample", templated_name):
            num_partitions = math.ceil(self.num_partitions_on_repartition / 4)

        if re.fullmatch(r"__splink__df_concat_with_tf", templated_name):
            num_partitions = math.ceil(self.num_partitions_on_repartition / 4)

        if re.fullmatch(r"|".join(names_to_repartition), templated_name):
            snowflake_df = snowflake_df.repartition(num_partitions)

        return snowflake_df

    def _break_lineage_and_repartition(self, snowflake_df, templated_name, physical_name):
        # snowflake_df = self._repartition_if_needed(snowflake_df, templated_name)

        return snowflake_df

        regex_to_persist = [
            r"__splink__df_comparison_vectors",
            r"__splink__df_concat_with_tf",
            r"__splink__df_predict",
            r"__splink__df_tf_.+",
            r"__splink__df_representatives.*",
            r"__splink__df_neighbours",
            r"__splink__df_connected_components_df",
        ]

        if re.fullmatch(r"|".join(regex_to_persist), templated_name):
            if self.break_lineage_method == "persist":
                snowflake_df = snowflake_df.persist()
                logger.debug(f"persisted {templated_name}")
            elif self.break_lineage_method == "checkpoint":
                snowflake_df = snowflake_df.checkpoint()
                logger.debug(f"Checkpointed {templated_name}")
            elif self.break_lineage_method == "parquet":
                checkpoint_dir = self._get_checkpoint_dir_path(snowflake_df)
                write_path = os.path.join(checkpoint_dir, physical_name)
                snowflake_df.write.mode("overwrite").parquet(write_path)
                snowflake_df = self.snowflake.read.parquet(write_path)
                logger.debug(f"Wrote {templated_name} to parquet")
            elif self.break_lineage_method == "delta_lake_files":
                checkpoint_dir = self._get_checkpoint_dir_path(snowflake_df)
                write_path = os.path.join(checkpoint_dir, physical_name)
                snowflake_df.write.mode("overwrite").format("delta").save()
                snowflake_df = self.snowflake.read.format("delta").load(write_path)
                logger.debug(f"Wrote {templated_name} to Delta files at {write_path}")
            elif self.break_lineage_method == "delta_lake_table":
                write_path = f"{self.splink_data_store}.{physical_name}"
                snowflake_df.write.mode("overwrite").saveAsTable(write_path)
                snowflake_df = self.snowflake.table(write_path)
                logger.debug(
                    f"Wrote {templated_name} to Delta Table at "
                    f"{self.splink_data_store}.{physical_name}"
                )
            else:
                raise ValueError(
                    f"Unknown break_lineage_method: {self.break_lineage_method}"
                )
        return snowflake_df

    def _set_default_break_lineage_method(self):
        # check to see if running in databricks and use delta lake tables
        # as break lineage method if nothing else specified.

        if self.in_databricks and not self.break_lineage_method:
            self.break_lineage_method = "delta_lake_table"
            logger.info(
                "Intermediate results will be written as Delta Lake tables at "
                f"{self.splink_data_store}."
            )

        # set non-databricks environment default method as parquet in case nothing else
        # specified.
        elif not self.break_lineage_method:
            self.break_lineage_method = "parquet"
