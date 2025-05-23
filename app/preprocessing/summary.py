from __future__ import annotations

import logging
import re
import time
from typing import Any

import pandas as pd
import tiktoken
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from app.llm.llm import SummaryLlm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummaryProccesing:
    def __init__(self):
        pass

    def add_column_count_tokens(self, df: pd.DataFrame, model:str='gpt-4o') -> pd.DataFrame:
        new_df = df.copy()
        tokenizer = tiktoken.encoding_for_model(model)
        new_df['number_of_tokenks'] = new_df['text'].apply(
            lambda x: len(tokenizer.encode(x)),
        )
        return new_df

    def custom_summary_document(self, document: str, llm:SummaryLlm) -> tuple[str, str]:
        template = llm.summary_prompt_template.strip()
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm.llm
        response = chain.invoke(document)
        return response.content, prompt.invoke(document).text

    def summarize_dataframe(self, df_in: pd.DataFrame, llm: SummaryLlm, sleep_time: int = 1)-> pd.DataFrame:
        prompts = []
        summaries = []
        df = df_in.copy()
        # Iterate over the rows using tqdm for progress tracking
        for text in tqdm(df['text'], desc='Summarizing texts'):
            try:
                # First attempt
                summary, prompt_text = self.custom_summary_document(text, llm)
            except Exception as e:
                logger.error(f"Error processing text: {e}. Retrying once...")
                time.sleep(30)  # Pause before retrying

                try:
                    # Retry if the first attempt fails
                    summary, prompt_text = self.custom_summary_document(text, llm)
                except Exception as e:
                    logger.error(f"Second attempt failed: {e}. Skipping this record.")
                    summary, prompt_text = None, None

            # Append results to the lists
            summaries.append(summary)
            prompts.append(prompt_text)

            # Sleep between each request
            time.sleep(sleep_time)

        df['prompt'] = prompts
        df['summary'] = summaries

        return df

    # Function to extract the key terms from the 'summary' column
    @staticmethod
    def _extract_key_terms(text):
        # Buscar el encabezado "Activos recomendados" con cualquier texto entre paréntesis o después de dos puntos
        match = re.search(r'Activos recomendados(?:\s*\([^)]+\))?\s*:?\**\n?', text, re.IGNORECASE)
        if not match:
            return None
        # Extraer el texto después del encabezado
        key_terms = text[match.end():]
        # Eliminar los guiones y números
        key_terms = re.sub(r'-\s*', '', key_terms)
        key_terms = re.sub(r'\d+\.', '', key_terms)
        # Tomar solo hasta el primer salto de línea doble (fin de la lista)
        key_terms = key_terms.split('\n\n')[0].strip()
        # Dividir en líneas y limpiar
        terms = [term.strip() for term in key_terms.split('\n') if term.strip()]
        return terms if terms else None

    def extract_key_terms_from_df(self, df_in, column_name='summary'):
        """
        Extracts key terms from a specified column in a DataFrame.
        """
        df = df_in.copy()
        # Check if the specified column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        # Apply the key term extraction function to the specified column
        df['key_terms'] = df['summary'].apply(self._extract_key_terms)

        return df
