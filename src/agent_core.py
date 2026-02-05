import os
import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.tools.custom_tools import get_current_datetime, get_pandas_tool

load_dotenv()

class DataAnalysisAgent:
    def __init__(self, df: pd.DataFrame = None, system_role: str = None):
        self.df = df
        self.chat_history = [] 
        
        # Configuração de Pastas
        self.output_dir = "outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Preparação das Tools
        self.tools = [get_current_datetime]
        
        if self.df is not None:
            pandas_tool = get_pandas_tool(self.df)
            if pandas_tool:
                self.tools.append(pandas_tool)

        # Configuração do Gemini 2.0 Flash Lite
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite-preview-02-05",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            safety_settings={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            }
        )
        
        role = system_role if system_role else "um Cientista de Dados Sênior"

        # PROMPT ATUALIZADO COM NUMPY
        self.prompt = PromptTemplate.from_template(
            f"""
            Você é {role}.
            
            SUAS FERRAMENTAS:
            {{tools}}
            
            ----------------------------------------------------------------
            DIRETRIZES DE USO:
            
            1. ANÁLISE DE DADOS:
               - Use a ferramenta `python_repl_ast`.
               - O dataframe é `df`.
               - O numpy está disponível como `np`.
               - Para cálculos matemáticos complexos (estatística avançada, logaritmos, transformações), PREFIRA usar `np` em vez de loops Python.
            
            2. VISUALIZAÇÃO (GRÁFICOS):
               - Salve sempre em: `{self.output_dir}/`
               - Exemplo: `plt.savefig('{self.output_dir}/analise_numpy.png')`
               - Use `plt.close()` ao final.
            
            3. GERAL:
               - Se perguntarem datas, use `get_current_datetime`.
            
            ----------------------------------------------------------------
            FORMATO DE PENSAMENTO (ReAct):
            
            Question: Pergunta do usuário
            Thought: O que fazer?
            Action: python_repl_ast
            Action Input: print(df.describe())
            Observation: ...
            ...
            Final Answer: Resposta final.
            
            Comece!
            
            Question: {{input}}
            Thought:{{agent_scratchpad}}
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
            history_text = "Histórico Recente:\n" + "\n".join([f"{r}: {c}" for r, c in self.chat_history[-2:]])
        
        full_input = f"{history_text}\nNova Pergunta: {user_input}" if history_text else user_input

        try:
            response = self.executor.invoke({"input": full_input})
            output = response['output']
            
            self.chat_history.append(("Usuário", user_input))
            self.chat_history.append(("Assistente", output))
            
            return output
            
        except Exception as e:
            return f"Erro no Agente: {str(e)}"