from __future__ import annotations

import logging
import os
import sys

import mlflow
import pandas as pd
from dotenv import load_dotenv
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain.evaluation.embedding_distance import EmbeddingDistance
from langchain.evaluation.embedding_distance import EmbeddingDistanceEvalChain
from langchain.evaluation.scoring import ScoreStringEvalChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# fmt: off
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# fmt: on

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class EvaluateModels:
    def __init__(
        self, llm_evaluator_name: str = None, model_embedding_name: str = None,
    ):
        if llm_evaluator_name is None:
            llm_evaluator_name = 'gpt-4o-mini'
        if model_embedding_name is None:
            model_embedding_name = 'text-embedding-ada-002'
        self.llm_evaluator_name = llm_evaluator_name
        self.embeddings_name = model_embedding_name
        self.llm_evaluator = ChatOpenAI(
            temperature=0, model=self.llm_evaluator_name, openai_api_key=OPENAI_API_KEY,
        )
        self.embeddings = OpenAIEmbeddings(
            model=self.embeddings_name, openai_api_key=OPENAI_API_KEY,
        )

    def run_evaluation(
        self,
        experiment_name: str,
        dataset_path: str,
        prompt_version: str,
        model_name: str,

    ):
        # Cargar el dataset
        dataset = pd.read_csv(dataset_path, sep=';')
        if dataset.empty:
            raise ValueError(f"El dataset está vacío: {dataset_path}")
        logging.info(f"📊 Dataset cargado: {dataset_path}")

        # Configurar el experimento
        self.set_experiment(experiment_name, dataset,prompt_version = prompt_version, model=model_name)


    def set_experiment(
        self,
        experiment_name: str,
        dataset: pd.DataFrame,
        prompt_version: str,
        model: str,
    ):
        mlflow.set_experiment(experiment_name)
        print(f"📊 Experimento MLflow: {experiment_name}")

        score_chain = ScoreStringEvalChain.from_llm(llm=self.llm_evaluator)
        embedding_chain = EmbeddingDistanceEvalChain(
            embeddings=self.embeddings, distance_metric=EmbeddingDistance.COSINE,
        )
        criteria = {
            'faithfulness': 'Is the summary accurate and consistent with the source text, without hallucinations or fabricated facts? If yes, respond Y. If no, respond N.',
            'relevance': 'Does the summary include the main information from the source text and omit insignificant details? If yes, respond Y. If no, respond N.',
            'conciseness': 'Is the summary concise and free of unnecessary repetition or verbosity? If yes, respond Y. If no, respond N.',
            'coherence': 'Is the summary well-structured, clear, and easy to follow? If yes, respond Y. If no, respond N.',
        }
        labeled_chain = LabeledCriteriaEvalChain.from_llm(
            llm=self.llm_evaluator, criteria=criteria,
        )

        for i, row in dataset.iterrows():
            video_id = row['video_id']
            channel_name = row['channel_name']
            reference = row['summary']  # reference summary
            prediccion = row['slm_summary']  # Model-generated summary
            input = row['slm_prompt']  # Prompt used for generation

            with mlflow.start_run(run_name=f"eval_q{i+1}", nested=True):
                # 1. LabeledCriteriaEvalChain
                labeled_eval = labeled_chain.evaluate_strings(
                    input=input, prediction=prediccion, reference=reference,
                )

                razonamiento = labeled_eval.get('reasoning', 'No reasoning')
                valor = labeled_eval.get(f"value", 'NaN')
                score = labeled_eval.get(f"score", 0)
                mlflow.log_metric('criterial_score', score)
                mlflow.log_param(f"criterial_value", valor)
                mlflow.log_text(razonamiento, f"criterial_reasoning.txt")

                # 2. EmbeddingDistanceEvalChain
                emb_eval = embedding_chain.evaluate_strings(
                    prediction=prediccion, reference=reference,
                )
                emb_score = emb_eval.get('score', 0)
                mlflow.log_metric('embedding_cosine_distance', emb_score)

                # 3. ScoreStringEvalChain
                str_eval = score_chain.evaluate_strings(
                    input=input, prediction=prediccion, reference=reference,
                )
                string_score = str_eval.get('score', 0)
                mlflow.log_metric('score', string_score)

                # Log información relevante
                mlflow.log_param('video_id', video_id)
                mlflow.log_param('channel_name', channel_name)
                mlflow.log_param('summary_reference', reference)
                mlflow.log_param('model_prediction', prediccion)
                mlflow.log_param('slm_prompt', input)
                mlflow.log_param('prompt_version', prompt_version)
                mlflow.log_param('llm_model', model)
                mlflow.log_param('llm_evaluator', self.llm_evaluator_name)
                mlflow.log_param('embedding_model', self.embeddings_name)


                logging.info(
                    f"[{i+1}] OK: Labeled:{labeled_eval} | Embedding Score:{emb_score} | StringScore:{string_score}",
                )
