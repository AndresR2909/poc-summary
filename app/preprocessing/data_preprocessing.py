"""procesamiento de datos"""
from __future__ import annotations

import logging
import re
from datetime import datetime

import pandas as pd


class DataPreprocessing:
    """Clase para el preprocesamiento de datos"""

    def executar_preprocesamiento_landing_a_raw(self, df_input: pd.DataFrame):
        """preprocesamiento de datos de landing a raw"""
        # fecha y hora actual utc
        if len(df_input) < 0:
            logging.info('No hay datos para preprocesar')
            raise ValueError('No hay datos para preprocesar')
        df_input.info()
        df_clean = self.preprocesar_nulos_duplicados(df_input)
        df_clean = self.eliminar_columnas(df_clean)
        df_clean.info()
        df_clean['last_update_date'] = datetime.now().strftime('%Y-%m-%d')
        df_clean['year'] = df_clean['publish_date'].apply(self.extraer_ano)
        df_clean.info()
        return df_clean

    def executar_preprocesamiento_raw_a_curated(self, df_input: pd.DataFrame):
        """preprocesamiento de datos de raw a curated"""
        if len(df_input) < 0:
            logging.info('No hay datos para preprocesar')
            raise ValueError('No hay datos para preprocesar')
        # aplicar transformaciones
        transformed_df = self.preprocesar_textos(df_input)
        transformed_df['publish_date'] = pd.to_datetime(transformed_df['publish_date'])

        return transformed_df

    def extraer_ano(self, publish_date):
        """extraer el año de la fecha de publicacion"""
        try:
            return int(publish_date[:4])
        except (ValueError, TypeError):
            return 0000

    def preprocesar_nulos_duplicados(self, df: pd.DataFrame) -> pd.DataFrame:
        """ preprocesar nulos y duplicados"""
        # eliminar duplicados
        df_clean = df.drop_duplicates(subset='video_id').reset_index(drop=True).copy()

        # Completar los datos nulos de 'duration' con los valores de 'total_length'
        df_clean['duration'] = df_clean['duration'].fillna(df_clean['total_length'])

        # eliminar registros con mas del 50% de columnas nulas
        df_clean = df_clean.dropna(thresh=len(df_clean) / 2, axis=1)

        return df_clean

    def eliminar_columnas(self, df: pd.DataFrame) -> pd.DataFrame:
        # eliminar columnas no usadas
        drop_columns = [
            'keywords',
            'description',
            'total_length',
            'total_views',
            'relativeDateText',
        ]

        df_colums = df.columns.to_list()

        drop_columns = [elemento for elemento in drop_columns if elemento in df_colums]

        df_clean = df.drop(columns=drop_columns)
        return df_clean

    def _limpiar_emoticones(self, texto: str) -> str:
        # Patrón para detectar emoticones
        patron_emoticones = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251\U0001f926-\U0001f937\U0001F9D1-\U0001F9DD]+'

        # Reemplazar los emoticones con un espacio en blanco
        texto_limpio = re.sub(patron_emoticones, '', texto)
        texto_limpio = texto_limpio.lower()

        return texto_limpio

    def _limpiar_texto(self, texto: str):
        texto = texto.lower()
        texto = re.sub(r'\s+', ' ', texto).strip()
        texto = texto.strip()

        return texto

    def preprocesar_textos(self, df_clean: pd.DataFrame) -> pd.DataFrame:

        df_clean['clean_title'] = df_clean['title'].apply(self._limpiar_emoticones)
        df_clean = df_clean.dropna(subset='caption_text_es', axis=0)
        df_clean['caption_text_es'] = (
            df_clean['clean_title'] + '. ' + df_clean['caption_text_es'].fillna(' ')
        )
        df_clean['text'] = df_clean['caption_text_es'].apply(self._limpiar_texto)
        df_clean = df_clean.drop(columns=['caption_text_es', 'title'])

        select_columns = [
            'channel_name',
            'video_id',
            'url',
            'publish_date',
            'duration',
            'last_update_date',
            'clean_title',
            'text',
            'year',
        ]
        df_select = df_clean[select_columns]
        df_select = df_select.rename(columns={'url': 'source', 'clean_title': 'title'})

        return df_select
