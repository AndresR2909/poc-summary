from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from app.preprocessing.data_preprocessing import DataPreprocessing
from app.storage.blob_storage import BlobStorage
from app.storage.data_lake_pandas import DataLakePandas
from app.storage.data_lake_storage import DataLakeStorage

load_dotenv()

bs_manager = BlobStorage()
dp = DataPreprocessing()
dl_pandas = DataLakePandas()

logging.basicConfig(level=logging.INFO)


class ETLProcess:
    """ Clase que contiene los métodos para ejecutar el proceso ETL """
    def run_etl(self):
        container_source = 'landing'
        path = 'youtube_transcripts/pending'
        sink_path = 'youtube_transcripts/processed/'
        timestamp = datetime.now()
        year = timestamp.year

        df_new, files_to_move = self.read_pending_files(container_source, path)

        if df_new.empty:
            logging.info('No hay archivos pendientes para procesar')
        else:

            logging.info(f"read {len(df_new)} registers from landing")

            df_new_raw = self.run_landing_to_raw(df_new, year)

            logging.info(f"processed {len(df_new_raw)} registers to raw")

            df_new_curated = self.run_raw_to_curated(df_new_raw, year)

            logging.info(f"processed {len(df_new_curated)} registers to curated")

            logging.info(f"Moving {len(files_to_move)} files to processed")
            for filename in files_to_move:
                print(filename.split('/')[-1])
                bs_manager.move_blob(
                    container_source,
                    filename,
                    container_source,
                    sink_path + filename.split('/')[-1],
                )


    def run_landing_to_raw(self, df_new, year):
        container_sink = 'raw'
        df_new_raw = dp.executar_preprocesamiento_landing_a_raw(df_new)
        logging.info(f"loading old data {year}")
        df_raw_old = dl_pandas.read_dataframe_from_parquet(
            container_name=container_sink,
            file_name=f"youtube_data/youtube_data_{year}",
            filter=None,
        )
        logging.info('Updating delta to raw')
        df_raw_to_save = pd.concat([df_raw_old, df_new_raw], ignore_index=True)
        logging.info('Saving df to raw')
        dl_pandas.save_dataframe_to_parquet_one_file(
            df=df_raw_to_save,
            container_name=container_sink,
            file_name='youtube_data',
            partition_col='year',
        )
        return df_new_raw

    def run_raw_to_curated(self, df_new_raw, year):
        df_new_curated = dp.executar_preprocesamiento_raw_a_curated(df_new_raw)
        df_new_curated = df_new_curated.sort_values('publish_date')
        logging.info(f"loading old data {year}")
        df_curated_old = dl_pandas.read_dataframe_from_parquet(
            container_name='curated',
            file_name=f"youtube_data/youtube_data_{year}",
            filter=None,
        )
        logging.info('Updating delta to curated')
        df_curated_to_save = pd.concat(
            [df_curated_old, df_new_curated], ignore_index=True,
        )
        logging.info('Saving df to curated')
        dl_pandas.save_dataframe_to_parquet_one_file(
            df=df_curated_to_save,
            container_name='curated',
            file_name='youtube_data',
            partition_col='year',
        )
        return df_new_curated

    @staticmethod
    def read_pending_files(container, path):
        files = bs_manager.list_files_from_folder_container(container, path)
        lista_dataframes = []
        files_to_move = []

        # Iterar sobre los archivos, leer cada archivo y agregar su DataFrame a la lista
        for file_info in files:
            filename = file_info['filename']
            file = bs_manager.download_file_from_blob(container, filename)
            if file is not None:
                df = pd.read_csv(file, sep=';')  # , index_col=0)
                lista_dataframes.append(df)
                files_to_move.append(filename)

        df_full = pd.concat(lista_dataframes, ignore_index=True)
        return df_full, files_to_move
