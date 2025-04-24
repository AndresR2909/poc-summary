# app/main_interface.py
from __future__ import annotations

import os
import sys

import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title='📚 Chatbot GenAI + Métricas', layout='wide')


# fmt: off
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.llm.llm import SummaryLlm
import altair as alt


# fmt: on


def generate_summary(text, model: list, prompt: list):
    """Genera un resumen utilizando un modelo SLM y una version de prompt"""
    llm_config = {
        'type': 'ollama',
        'model': model,
        'base_url': 'http://localhost:11434',
    }
    summary_llm = SummaryLlm(config=llm_config, prompt_name=prompt)
    summary = summary_llm.summarize(text)
    return summary


def load_text_artifact(run_id, artifact_path):
    """Funcion para cargar artefactos txt de razonamiento criterios de evaluacion"""
    # Descargamos el contenido del artefacto
    artifact_content = client.download_artifacts(run_id, artifact_path)

    # Leemos el contenido del archivo
    with open(artifact_content) as file:
        artifact_text = file.read()

    return artifact_text


# Barra lateral de opciones
modo = st.sidebar.radio(
    'Selecciona una vista:',
    [
        '🤖 Chatbot',
        '📊 Métricas',
        '📊 Razonamientos evaluación',
    ],
)

###################################################
# Sección de Chatbot
###################################################
if modo == '🤖 Chatbot':
    st.title('🤖 Asistente de reporte videos')

    pregunta = st.text_input('Ingresa url del video para generar reporte:')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # if pregunta:
    #     with st.spinner('Consultando documentos...'):
    #         result = chain.invoke(
    #             {'question': pregunta, 'chat_history': st.session_state.chat_history},
    #         )
    #         st.session_state.chat_history.append((pregunta, result['answer']))

    if st.session_state.chat_history:
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**👤 Usuario:** {q}")
            st.markdown(f"**🤖 Bot:** {a}")
            st.markdown('---')

###################################################
# Sección de Métricas
###################################################
elif modo == '📊 Métricas':
    st.title('📈 Resultados de Evaluación')

    client = mlflow.tracking.MlflowClient()
    # Cargamos experimentos que comiencen con "eval_"
    experiments = [
        exp for exp in client.search_experiments() if exp.name.startswith('report_summary')
    ]

    if not experiments:
        st.warning('No se encontraron experimentos de evaluación.')
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox('Selecciona un experimento:', exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[
            'start_time DESC',
        ],
        max_results = 20000,
    )

    if not runs:
        st.warning('No hay ejecuciones registradas.')
        st.stop()

    # Recolectamos datos de cada run
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        artifacts = client.list_artifacts(run.info.run_id)
        list_artifacts = [artifact for artifact in artifacts]
        dict_metrics = {
            #'run_ID': run.info.run_id,
            'video_id': params.get('video_id'),
            'channel_name': params.get('channel_name'),
            'prompt_version': params.get('prompt_version'),
            'model': params.get('llm_model'),
            # Métricas de evaluación
            'criterial_score': metrics.get('criterial_score', None),
            'embedding_cosine_distance': metrics.get('embedding_cosine_distance', None),
            'score': metrics.get('score', None),
        }
        data.append(dict_metrics)

    # Creamos un dataframe con todos los datos
    df = pd.DataFrame(data)

    # Datos de test
    st.subheader('Dataset de test')

    df_grouped = (
        df.drop(columns=['criterial_score','embedding_cosine_distance','model', 'prompt_version','score'])
        .groupby(['channel_name'])
        .count()
        .reset_index()
    )
    st.dataframe(df_grouped)

    df_grouped = (
        df.drop(columns=['criterial_score','embedding_cosine_distance','score'])
        .groupby(['model', 'prompt_version'])
        .count()
        .reset_index()
    )
    st.dataframe(df_grouped)

    # Filtrar y agrupar dataset por Chunk Size y Prompt, sacar promedio del resto de columnas
    df_grouped = (
        df.drop(columns=['video_id', 'channel_name'])
        .groupby(['model', 'prompt_version'])
        .mean()
        .reset_index()
    )

    # Selección de criterios a mostrar en un gráfico
    st.subheader('criterios de evaluación agrupados por modelo y version de prompt')

    st.dataframe(df_grouped)

    # Selección de criterios a mostrar en un gráfico
    st.subheader('Comparar metricas de evaluación por modelo y version de prompt')

    # Posibles métricas disponibles
    metric_choices = [
        'criterial_score',
        'embedding_cosine_distance',
        'score',
    ]
    selected_metrics = st.multiselect(
        'Selecciona los criterios que deseas comparar',
        metric_choices,
        # Por defecto mostrar correctness vs hallucination
        default=['score', 'embedding_cosine_distance','criterial_score'],
    )

    if selected_metrics:
        # Agrupamos por Prompt y Chunk Size para mostrar promedios
        grouped = (
            df.groupby(['model', 'prompt_version'])
            .agg({metric: 'mean' for metric in selected_metrics})
            .reset_index()
        )

        grouped['config'] = (
            grouped['model'] + ' -> ' + grouped['prompt_version'].astype(str)
        )

        # Mostramos cada métrica seleccionada en un gráfico independiente
        if selected_metrics:
            st.subheader(
            'Promedio de métricas seleccionadas por configuración (gráficas independientes)',
            )


            for metric in selected_metrics:
                st.markdown(
                    f"<h3 style='text-align:center; text-transform:uppercase; font-size:2em;'>{metric}</h3>",
                    unsafe_allow_html=True,
                )
                chart = (
                    alt.Chart(grouped)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            'config:N',
                            axis=alt.Axis(labelAngle=-45, title=None),
                        ),
                        y=alt.Y(f"{metric}:Q", axis=alt.Axis(title=f"{metric}")),
                        color=alt.Color('config:N', legend=alt.Legend(title='Modelo')),
                        tooltip=['config', metric],
                        text=alt.Text(f"{metric}:Q", format='.2f'),
                    )
                    .properties(
                        width=alt.Step(60),
                        height=300,
                    )
                )

                # Agregar etiquetas encima de cada barra
                text = chart.mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-2,
                    fontSize=12,
                ).encode(
                    text=alt.Text(f"{metric}:Q", format='.2f'),
                )

                st.altair_chart(chart + text, use_container_width=True)


###################################################
# Sección de Artefactos
###################################################
elif modo == '📊 Artefactos Razonamiento evaluación':
    st.title('📈 Razonamientos de evaluacion de preguntas por ejecucion')

    # Posibles métricas disponibles
    metric_choices = [
        'criterial_score',
    ]

    client = mlflow.tracking.MlflowClient()
    # Cargamos experimentos que comiencen con "eval_"
    experiments = [
        exp for exp in client.search_experiments() if exp.name.startswith('report_summary')
    ]

    if not experiments:
        st.warning('No se encontraron experimentos de evaluación.')
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox('Selecciona un experimento:', exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[
            'start_time DESC',
        ],
    )

    if not runs:
        st.warning('No hay ejecuciones registradas.')
        st.stop()

    # Recolectamos datos de cada run
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        artifacts = client.list_artifacts(run.info.run_id)
        list_artifacts = [artifact for artifact in artifacts]
        dict_metrics = {
            'run_ID': run.info.run_id,
            'video_id': params.get('video_id'),
            'channel_name': params.get('channel_name'),
            'prompt_version': params.get('prompt_version'),
            'model': params.get('llm_model'),
            'model_prediction': params.get('model_prediction'),
            'summary_reference': params.get('summary_reference'),
            # Métricas de evaluación
            'criterial_score': metrics.get('criterial_score', None),
            'embedding_cosine_distance': metrics.get('embedding_cosine_distance', None),
            'score': metrics.get('score', None),
        }
        text_artifacts = {}
        if len(list_artifacts) > 0:
            for artifact in list_artifacts:
                path = artifact.path
                name = path.split('.txt')[0]
                text_artifacts[name] = load_text_artifact(
                    run.info.run_id,
                    path,
                )
        else:
            text_artifacts = {
                'criterial_reasoning': None,
            }

        dict_metrics.update(text_artifacts)
        data.append(dict_metrics)

    # Creamos un dataframe con todos los datos
    df = pd.DataFrame(data)

    # Mostrar razonamientos
    st.subheader('Razonamientos de evaluacion de criterios')

    # Selección de Run ID para filtrar razonamientos
    video_ids = df['video_id'].unique()
    selected_run_id = st.selectbox(
        'Selecciona un video_id para ver los razonamientos:',
        video_ids,
    )
    models = df['model'].unique()

    selected_pregunta = st.selectbox(
        'Selecciona un modelo para ver los razonamientos:',
        models,
    )

    # Filtramos el DataFrame por el Run ID y pregunta seleccionado
    filtered_df = df[
        (df['video_id'] == selected_run_id) & (df['model'] == selected_pregunta)
    ]

    # Mostramos los razonamientos del Run ID y Pregunta seleccionados

    st.markdown(f"**model_prediction:** {filtered_df['model_prediction'].values}")
    st.markdown(f"**summary_reference:** {filtered_df['summary_reference'].values}")
    # Mostramos los razonamientos
    st.markdown('**Razonamientos de evaluación:**')
    st.markdown(
        f"- **criterial_reasoning**: {filtered_df['criterial_reasoning'].values}",
    )
