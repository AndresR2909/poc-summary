from __future__ import annotations

import os
from typing import Dict
from typing import List
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

default_config = {
    'type': 'ollama',
    'model': 'phi4:latest',
    'base_url': 'http://localhost:11434',
}


class Llm:
    def __init__(self, config: dict = default_config):
        self.type = config.get('type')
        if self.type == 'ollama':
            self.base_url = config.get('base_url')
            self.model = config.get('model')
            self.llm = ChatOllama(base_url=self.base_url, model=self.model)
        elif self.type == 'openai':
            self.api_key = config.get('api_key')
            self.model = config.get('model', 'gpt-4o-mini')
            self.temperature = config.get('temperature', 0)
            self.max_completion_tokens = config.get('max_tokens')
            self.model = config.get('model')
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                timeout=120,
                max_retries=2,
                # api_key="...",
                # base_url="...",
            )
        else:
            raise ValueError(
                'Unsupported LLM type. Supported types are: ollama, openai',
            )

    def get_llm(self):
        return self.llm

    def load_prompt(self, prompt_path: str):
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
        with open(prompt_path) as f:
            prompt_text = f.read()
        return prompt_text


class SummaryLlm(Llm):
    def __init__(self, config: dict = default_config):
        super().__init__(config)

        self.summary_prompt_template = self.load_prompt(
            prompt_path='app/prompts/v1_summary_expert.txt',
        )

    def summarize(self, context):
        template = PromptTemplate(
            template=self.summary_prompt_template,
            input_variables=['context'],
        )

        qna_chain = template | self.llm | StrOutputParser()
        return qna_chain.invoke({'context': context})
