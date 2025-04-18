from __future__ import annotations

import logging
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.ERROR)

connection_string = os.environ.get('DATALAKE_CONNECTION_STRING')
sa = os.environ.get('SA_DATALAKE')


class DataLakePandas:
    def __init__(self) -> None:
        self.storage_options = {'connection_string': connection_string}
        self.url = f"abfs://{sa}.dfs.core.windows.net"
        pass

    def read_dataframe_from_csv(
        self, container_name: str, file_name: str, sep: str = ';',
    ) -> pd.DataFrame:
        try:
            path = self.url + f"/{container_name}/{file_name}.csv"
            # read data file
            df = pd.read_csv(path, sep=sep, storage_options=self.storage_options)
            return df

        except Exception as e:
            logging.error(f"Error reading file: {e}")
            return None

    def read_dataframe_from_parquet(
        self, container_name: str, file_name: str, filter: list | None,
    ) -> pd.DataFrame:
        try:
            path = self.url + f"/{container_name}/{file_name}"
            # read data file
            df = pd.read_parquet(
                path, storage_options=self.storage_options, filters=filter, engine='pyarrow',
            )
            return df

        except Exception as e:
            logging.error(f"Error reading file: {e}")
            return None

    def save_dataframe_to_parquet(
        self,
        df: pd.DataFrame,
        container_name: str,
        file_name: str,
        partition_col: str = 'year',
    ):
        try:
            path = self.url + f"/{container_name}/{file_name}"
            # write data file
            df.to_parquet(
                path,
                index=False,
                storage_options=self.storage_options,
                partition_cols=[partition_col],
                mode='',
            )

        except Exception as e:
            logging.error(f"Error saving dataframe into file: {e}")

    def save_dataframe_to_csv(
        self, df: pd.DataFrame, container_name: str, file_name: str, sep: str = ';',
    ):
        try:
            path = self.url + f"/{container_name}/{file_name}.csv"
            # write data file
            df.to_csv(path, sep=sep, index=False, storage_options=self.storage_options)

        except Exception as e:
            logging.error(f"Error saving dataframe into file: {e}")

    def save_dataframe_to_parquet_one_file(
        self,
        df: pd.DataFrame,
        container_name: str,
        file_name: str,
        partition_col: str = 'year',
    ):
        try:
            particiones = df[partition_col].unique()
            print(particiones)
            for filter_particion in particiones:

                df_particion = df[df[partition_col] == filter_particion]

                path = (
                    self.url
                    + f"/{container_name}/{file_name}/{file_name}_{filter_particion}"
                )
                print(path)
                # write data file
                df_particion.to_parquet(
                    path, index=False,
                    storage_options=self.storage_options,
                    engine='pyarrow',

                )

        except Exception as e:
            logging.error(f"Error saving dataframe into file: {e}")
