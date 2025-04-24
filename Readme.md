# POC Summary

Este proyecto contiene el código y notebooks que documentan las etapas de procesamiento, generación de datasets con resúmenes de transcripciones de videos y evaluación de resultados con SLMs para generar reportes estructurados a partir de videos de YouTube sobre análisis técnico y fundamental de trading.

---

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Flujo de Trabajo](#flujo-de-trabajo)
- [Notebooks Principales](#notebooks-principales)
- [Modelos Finetuneados](#modelos-finetuneados)
- [Ejecución de la Interfaz](#ejecución-de-la-interfaz)
- [Seguimiento de Experimentos con MLflow](#seguimiento-de-experimentos-con-mlflow)

---

## Descripción General

El objetivo es ajustar modelos SLM (Small Language Models) como Llama 3.2-1B y 3B para generar resúmenes precisos y contextuales de transcripciones de videos, usando instrucciones específicas y formatos de reporte. El pipeline incluye preprocesamiento, generación de datasets, fine-tuning, evaluación y despliegue.

---

## Estructura del Proyecto

```
app/
  main_interface.py
  llm/
  preprocessing/
  prompts/
  storage/
data/
  ...
notebooks/
  run_etl.ipynb
  create_summaries.ipynb
  create_summaries_v2.ipynb
  summary_with_slm.ipynb
  train_deploy_litgpt.ipynb
  evaluate_summary.ipynb
  test_mlflow.ipynb
requirements.txt
Readme.md
...
```

---

## Flujo de Trabajo

1. **Preprocesamiento y ETL:** Limpieza y normalización de transcripciones.
2. **Generación de Datasets:** Creación de resúmenes con LLMs grandes (GPT-4.1, GPT-4o) para usar como referencia y entrenamiento.
3. **Fine-Tuning:** Ajuste supervisado de SLMs con los datasets generados.
4. **Evaluación:** Comparación de resúmenes generados por SLMs vs. referencia usando métricas automáticas y evaluación con otro LLM.

 ![evaluation](image.png)

5. **Despliegue y Uso:** Interfaz para generar resúmenes y tablero MLflow para seguimiento de experimentos.

---

## Notebooks Principales

- [`run_etl.ipynb`](notebooks/run_etl.ipynb):
  **Funcionalidad:** Preprocesamiento de datos de entrada, limpieza de textos y preparación del dataset.

- [`create_summaries.ipynb`](notebooks/create_summaries.ipynb) y [`create_summaries_v2.ipynb`](notebooks/create_summaries_v2.ipynb):
  **Funcionalidad:** Generación de resúmenes de transcripciones usando modelos LLM fundacionales (GPT-4o, GPT-4.1) para destilación de conocimiento.

- [`summary_with_slm.ipynb`](notebooks/summary_with_slm.ipynb):
  **Funcionalidad:** Generación de resúmenes con diferentes SLMs, prompts (zero/one shot) y modelos finetuneados.

- [`train_deploy_litgpt.ipynb`](notebooks/train_deploy_litgpt.ipynb):
  **Funcionalidad:** Fine-tuning supervisado de modelos Llama 3.2-1B y 3B usando LoRA y QLoRA, conversión de datasets a formato JSON, merge de pesos y despliegue.

['Repo espacio de trabajo lightning.ai']()

- [`evaluate_summary.ipynb`](notebooks/evaluate_summary.ipynb):
  **Funcionalidad:** Evaluación de resúmenes generados por SLMs usando métricas automáticas, embeddings y evaluación con otro LLM (GPT-4.1).

- [`test_mlflow.ipynb`](notebooks/test_mlflow.ipynb):
  **Funcionalidad:** Revisión y comparación de resultados de experimentos registrados en MLflow.

---

## Modelos Finetuneados

- [llama-3.2-1b-finetuned_v5](https://huggingface.co/AndresR2909/hf-llama-3.2-1b-finetuned_v5)
- [llama-3.2-3b-finetuned_bnb_nf4](https://huggingface.co/AndresR2909/hf-llama-3.2-3b-finetuned_bnb_nf4)
- [llama-3.2-3b-finetuned_bnb_nf4_v2](https://huggingface.co/AndresR2909/hf-llama-3.2-3b-finetuned_bnb_nf4_v2)

Datasets en HuggingFace:
- [youtube_transcriptions_summaries_2025_gpt4.1](https://huggingface.co/datasets/AndresR2909/youtube_transcriptions_summaries_2025_gpt4.1)
- [youtube_transcriptions_summaries_2025_gpt4o](https://huggingface.co/datasets/AndresR2909/youtube_transcriptions_summaries_2025_gpt4o/)

---

## Ejecución de la Interfaz

Para correr la interfaz que permite generar resúmenes y obtener reportes de métricas:

```bash
streamlit run app/main_interface.py
```

![dataset test](image-1.png)

![metricas evaluacion](image-2.png)

![Score](image-6.png)

![grafica coseno](image-4.png)

![criterial score](image-5.png)

```bash
streamlit run app/main_interface.py
```



---

## Seguimiento de Experimentos con MLflow

Para lanzar el tablero de MLflow y visualizar los experimentos:

```bash
mlflow ui --port 5000
```
