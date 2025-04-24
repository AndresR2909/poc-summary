# Notebooks de POC Summary

Este proyecto contiene el codigo y notebooks que documentan las etapas del procesamiento, resumen y evaluación de SLMs para generar un resumen con un formato especifico de reporte de transcripciones de videos de youtube donde se realizan analisis tecnicos y fundamentales de trading.

## Índice de Notebooks

- [run_etl.ipynb](notebook/run_etl.ipynb):
  **Funcionalidad:**
  Demuestra el proceso de preprocesamiento de los datos de entrada. Incluye limpieza de textos, normalización y preparación del dataset con los datos ingestados de las trnacripciones de los videos. Se realiza el movimiento de archivos desde la zona landing a la zona curated del datalake.

- [create_summaries.ipynb](notebook/create_summaries.ipynb):
- [create_summaries_v2.ipynb](notebooks/create_summaries_v2.ipynb):
  **Funcionalidad:**
 crear dataset de resumenes de transcripciones de videos usando modelo fundacional LLM gpt4o y gpt4.1(Destilación de conocimiento: usado para hacer transferencia de conocimiento a SLM)

[dataset v1 de entrenaminto en huggingface](hhttps://huggingface.co/datasets/AndresR2909/youtube_transcriptions_summaries_2025_gpt4o/)
[dataset  v2 de entrenaminto(90%) y test(10%) en huggingface](https://huggingface.co/datasets/AndresR2909/youtube_transcriptions_summaries_2025_gpt4.1/)

La destilación de conocimiento implica transferir los aprendizajes de un "modelo maestro" gpt4.1 preentrenado a un "modelo estudiante" en este caso llama 3.2 instruct de 1B y 3B de parametros.


- [train_deploy_litgpt.ipynb](notebook/train_deploy_litgpt.ipynb):
  **Funcionalidad:**

proceso de **Supervised Fine-Tuning (SFT)** de un modelo Llama 3.2-3B-Instruct, adaptándolo para tareas de resumen automático a partir de transcripciones de videos de YouTube. Aprenderás a preparar datos reales, ajustar el modelo y evaluar su desempeño en generación de resúmenes.

### Objetivo:
Ajustar el modelo **Llama 3.2-3B-Instruct** para que genere resúmenes precisos y contextuales a partir de transcripciones de videos, usando instrucciones específicas.

Convertir dataset creado en HuginFace a Formato json de entrada, instruccion, salida para usarlo en el finetuning del modelo.
Finetuning de modelos SLM llama 3.2 intruct 1b y 3b usando Lora y Qlora
Merge Lora Weights
Deploy
Subir modelos a HuggingFace (para desplegarlo con ollama y poderlo usar en aplicacion local)

[llama-3.2-3b-finetuned_bnb_nf4](https://huggingface.co/AndresR2909/hf-llama-3.2-3b-finetuned_bnb_nf4)

[llama-3.2-1b-finetuned_v5](https://huggingface.co/AndresR2909/hf-llama-3.2-1b-finetuned_v5)


[llama-3.2-3b-finetuned_bnb_nf4_v2](https://huggingface.co/AndresR2909/hf-llama-3.2-3b-finetuned_bnb_nf4_v2)

- [summary_with_slm.ipynb](notebook/summary_with_slm.ipynb):
**Funcionalidad:**
crear resumenes con diferentes SLMs, prompts zero an one shot y con los SLMs finetuneados

- [evaluate_summary.ipynb](notebook/evaluate_summary.ipynb):
**Funcionalidad:**
evaluar los resultados con diferentes modelos usando metricas basadas en embedings y score con otro LLM como evaluador(GPT4.1) y usando el framework mlflow para el tracking de los experimentos

- [test_mlflow.ipynb](notebook/test_mlflow.ipynb):
**Funcionalidad:**
revisar resultados de run experimentos cargado parametros y logs para comparar



## Ejecución de la interfaz

Para correr la interfaz que permite generar resúmenes y obtener reportes de métricas, ejecuta el siguiente comando en la terminal:

```bash
streamlit run app/main_interface.py
```
## Ejecucion de tablero de mlflow para seguimiento de experimentos
```bash
mlflow ui --port 5000
```
