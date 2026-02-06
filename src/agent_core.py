import os
import warnings
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# --- FILTROS DE AVISO ---
warnings.filterwarnings("ignore")

# --- IMPORTAÇÃO ROBUSTA DO AGENTE ---
try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    try:
        from langchain.agents.agent import AgentExecutor
        from langchain.agents.react.agent import create_react_agent
    except ImportError:
        from langchain.agents import AgentExecutor
        from langchain.agents import create_react_agent

from src.dataset_manager import DatasetManager
from src.tools.custom_tools import get_analysis_tools

load_dotenv()

class DataAnalysisAgent:
    def __init__(self, system_role: str = None):
        # 1. Inicializa Embeddings Locais (HuggingFace)
        self.dataset_manager = DatasetManager()
        self.chat_history = [] 
        
        self.output_dir = "outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 2. Carrega Tools
        self.tools = get_analysis_tools(self.dataset_manager)

        # 3. Configuração do Gemini
        # REMOVEMOS O 'safety_settings' PARA EVITAR O ERRO DE VALIDAÇÃO.
        # O modelo usará os filtros padrão (Default), o que permite o código rodar.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        role = system_role if system_role else "um Agente Autônomo de Dados"

        # 4. Prompt
        self.prompt = PromptTemplate.from_template(
            f"""
            Você é {role}.
            
            FERRAMENTAS:
            {{tools}}
            
            NOMES: {{tool_names}}
            
            ----------------------------------------------------------------
            REGRAS:
            1. DADOS: Se `df` for None, use `load_relevant_dataset`. Se tiver dados, use `python_repl_ast`.
            2. CÓDIGO: Use `print()` para ver resultados.
            3. GRÁFICOS: Salve em `{self.output_dir}/nome.png` e limpe com `plt.close()`.
            
            ----------------------------------------------------------------
            Thought: {{agent_scratchpad}}
            """
        )

        agent = create_react_agent(self.llm, self.tools, self.prompt)
        
        self.executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=20
        )

    def process_message(self, user_input: str):
        history_text = ""
        if self.chat_history:
            history_text = "Histórico:\n" + "\n".join([f"{r}: {c}" for r, c in self.chat_history[-2:]])
        
        full_input = f"{history_text}\nUsuário: {user_input}" if history_text else user_input

        try:
            response = self.executor.invoke({"input": full_input})
            return response['output']
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                return "⚠️ Cota excedida. Aguarde um instante."
            return f"Erro: {error_msg}"