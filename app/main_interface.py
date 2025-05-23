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
from app.youtube_ingest import YoutubeIngest


# fmt: on
client = mlflow.tracking.MlflowClient()


yt = YoutubeIngest()

experiments = [
    exp
    for exp in client.search_experiments()
    if exp.name.startswith('report_summary')
]
exp_names = [exp.name for exp in experiments]

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


def generate_summary_from_url(url_video: str, model: str, prompt: str):
    """Genera un resumen utilizando un modelo SLM y una version de prompt"""
    from app.llm.llm import SummaryLlm

    llm_config = {
        'type': 'ollama',
        'model': model,
        'base_url': 'http://localhost:11434',
    }
    # cargar llm con prompt
    summary_llm = SummaryLlm(config=llm_config, prompt_name=prompt)
    # sacar id del video
    sel_video_id = yt.extract_video_id_from_url(url_video)
    # optener trasncripcion
    transcription = yt.get_transcript_by_id(video_id=sel_video_id)
    # generar reporte
    report = summary_llm.summarize(transcription)
    return report


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
        '🤖 Reporte video',
        '📊 Métricas',
        '📊 Razonamientos evaluación',
    ],
)

###################################################
# Sección Reporte video
###################################################
if modo == '🤖 Reporte video':
    st.title('🤖 Asistente de reporte videos')
    sel_columns = ['videoId', 'title', 'publishTime', 'videoUrl', 'duration']
    # Listar ultimos videos de los canales en yt.channels

    #daysback = st.selectbox("Selecciona dias de busqueda :", ["1", "7", "14"])
    #int_daysback = int(daysback)
    st.subheader(f"Videos disponibles por canal (últimos 7 días)")

    # Usar st.session_state para cachear los resultados y evitar recargas innecesarias
    if 'videos_metadata' not in st.session_state:
        st.session_state['videos_metadata'] = {}

    if st.button('Cargar videos recientes de los canales'):
        st.session_state['videos_metadata'] = {}
        for channel_name in yt.channels:
            df = pd.DataFrame(
                yt.get_last_videos_metadata_from_channels(
                    channel_name, daysback=7,
                ),
            )
            if len(df) > 0:
                st.session_state['videos_metadata'][channel_name] = df[sel_columns]

    # Mostrar los datos si ya están cargados
    if st.session_state['videos_metadata']:
        for channel_name, df in st.session_state['videos_metadata'].items():
            st.markdown(f"**{channel_name}**")
            st.dataframe(df)

    # Entrada de enlace de YouTube o selección desde la tabla
    youtube_url = st.text_input('Ingresa el enlace del video de YouTube:')

    # Si hay videos cargados, permitir seleccionar uno de la tabla
    all_video_urls = []
    for df in st.session_state['videos_metadata'].values():
        if 'videoUrl' in df.columns:
            all_video_urls.extend(df['videoUrl'].tolist())

    if all_video_urls:
        selected_url = st.selectbox(
            'O selecciona un video de la lista:',
            [''] + all_video_urls,
            index=0,
        )
        # Si selecciona uno de la lista, sobrescribe el valor del input manual
        if selected_url:
            youtube_url = selected_url

    # Selección de modelo y prompt
    model = st.selectbox(
        'Selecciona el modelo LLM:',
        [
            'phi4:latest',
            'hf.co/AndresR2909/unsloth_Meta-Llama-3.1-8B-Instruct-bnb-4bit_gguf:Q8_0',
            'hf.co/AndresR2909/llama-3.2-3b-finetuned_qlora_bnb_nf4_v2-gguf_q8_0:latest',
            'hf.co/AndresR2909/unsloth_Meta-Llama-3.1-8B-Instruct-bnb-4bit_gguf_v3:Q8_0',
        ],
    )
    prompt = st.selectbox(
        'Selecciona la versión del prompt:',
        ['v3_summary_expert', 'v1_summary_expert_one_shot'],
    )
    # model = 'phi4:latest'
    # prompt = 'v1_summary_expert'

    if st.button('Generar resumen'):
        if not youtube_url:
            st.warning('Por favor ingresa un enlace de YouTube.')
        else:
            try:
                with st.spinner('Generando resumen...'):
                    resumen = generate_summary_from_url(youtube_url, model, prompt)
                    st.subheader('Resumen generado:')
                    st.write(resumen)
            except Exception as e:
                st.error(f"Error al extraer y generar reporte del video: {e}")
                resumen = None
elif modo == '📊 Métricas':
    st.title('📈 Resultados de Evaluación')
    # Cargamos experimentos que comiencen con "eval_"

    if not experiments:
        st.warning('No se encontraron experimentos de evaluación.')
        st.stop()


    selected_exp = st.selectbox('Selecciona un experimento:', exp_names)

    experiment = client.get_experiment_by_name(selected_exp)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[
            'start_time DESC',
        ],
        max_results=20000,
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
    sel_models = [
        'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit_gguf2_Q4_k_m',
        'phi4_latest',
        'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit_gguf3_Q4_k_m',
        'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit_gguf3_Q8_0',
        'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit_gguf2_Q8_0',
        'llama3_1_8b_instruct_fp16',
        'llama3_2_3b_instruct_fp16',
        'gpt_4o_mini',
        'llama_3_2_3b_finetuned_qlora_bnb_nf42_gguf_q8_0_latest',
    ]

    df_sel = df[df['model'].isin(sel_models)]

    # Diccionario de reemplazo de nombres de modelos
    model_name_map = {
    'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit_gguf2_Q4_k_m':'finetune_qlora_unsloth_llama_3_1_8B_Instruct_bnb_4bit_v2_Q4_k_m',
    'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit_gguf3_Q4_k_m':'finetune_qlora_unsloth_llama_3_1_8B_Instruct_bnb_4bit_v3_Q4_k_m',
    'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit_gguf3_Q8_0':'finetune_qlora_unsloth_llama_3_1_8B_Instruct_bnb_4bit_v3_Q8_0',
    'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit_gguf2_Q8_0':'finetune_qlora_unsloth_llama_3_1_8B_Instruct_bnb_4bit_v2_Q8_0',
    'llama_3_2_3b_finetuned_qlora_bnb_nf42_gguf_q8_0_latest': 'finetune_qlora_litgpt_llama_3_2_3b_bnb_nf4_v2_q8_0',
    }

    # Reemplazar en la columna 'model' de df_sel
    df_sel['model'] = df_sel['model'].replace(model_name_map)

    # Datos de test
    st.subheader('Dataset de test')

    df_grouped = (
        df_sel.drop(
            columns=[
                'criterial_score',
                'embedding_cosine_distance',
                'model',
                'prompt_version',
                'score',
            ],
        )
        .groupby(['channel_name'])
        .count()
        .reset_index()
    )
    st.dataframe(df_grouped)

    df_grouped = (
        df.drop(columns=['criterial_score', 'embedding_cosine_distance', 'score'])
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
        default=['score', 'embedding_cosine_distance', 'criterial_score'],
    )

    if selected_metrics:
        # Agrupamos por Prompt y Chunk Size para mostrar promedios
        grouped = (
            df_sel.groupby(['model', 'prompt_version'])
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

    # client = mlflow.tracking.MlflowClient()
    # Cargamos experimentos que comiencen con "eval_"
    experiments = [
        exp
        for exp in client.search_experiments()
        if exp.name.startswith('report_summary')
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
