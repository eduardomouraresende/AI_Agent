import os
import pandas as pd
from dotenv import load_dotenv

# MUDANÇA: Importamos a biblioteca do Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

class DataAnalysisAgent:
    def __init__(self, df: pd.DataFrame, system_role: str = None):
        self.df = df
        self.chat_history = [] 
        
        # CONFIGURAÇÃO DO GEMINI
        # Usamos o 'gemini-1.5-flash' por ser rápido e ter bom limite no free tier.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            # O Gemini as vezes bloqueia respostas por segurança (ex: dados médicos).
            # Ajustamos para não bloquear análise de dados inofensiva.
            safety_settings={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            }
        )
        
        self.output_dir = "outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        role = system_role if system_role else "um Cientista de Dados Sênior"
        
        # PROMPT (Adaptado para funcionar bem com o Gemini)
        self.prefix_prompt = f"""
        Você é {role}.
        
        FERRAMENTA:
        - Use APENAS a ferramenta: `python_repl_ast`.
        
        PROTOCOLO OBRIGATÓRIO (FORMATO DE PENSAMENTO):
        Para usar o Python, você DEVE seguir estritamente este padrão:
        
        Thought: Preciso verificar X...
        Action: python_repl_ast
        Action Input: print(df.head())
        
        REGRAS DE DADOS:
        1. Sempre use `print()` para ver o resultado do código. O Gemini não "vê" o retorno se não houver print.
        2. Para buscar texto: `df[df['col'].str.contains('Termo', case=False, na=False)]`.
        
        REGRAS DE GRÁFICOS:
        1. Pergunte antes de gerar.
        2. Se autorizado: Salve em `{self.output_dir}/nome_descritivo.png`.
        3. Nunca use `plt.show()`.

        Responda em Português do Brasil.
        """

        self.executor = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="zero-shot-react-description",
            max_iterations=30, 
            prefix=self.prefix_prompt,
            agent_executor_kwargs={
                "handle_parsing_errors": True 
            }
        )

    def process_message(self, user_input: str):
        # Histórico limitado a 2 trocas
        history_text = ""
        if self.chat_history:
            history_text = "Última conversa:\n"
            for role, content in self.chat_history[-2:]: 
                history_text += f"{role}: {content}\n"
            history_text += "\n"

        final_prompt = f"""
        {history_text}
        Pergunta atual: {user_input}
        
        Lembre-se: Use a ferramenta 'python_repl_ast' para executar código.
        """
        
        try:
            response = self.executor.invoke(final_prompt)
            output = response['output']
            
            self.chat_history.append(("Usuário", user_input))
            self.chat_history.append(("Assistente", output))
            
            return output
            
        except Exception as e:
            # Tratamento de erros comuns do Google
            error_msg = str(e)
            if "429" in error_msg or "Resource exhausted" in error_msg:
                return "⚠️ Limite de uso do Gemini atingido (Rate Limit). Aguarde um minuto."
            if "safety" in error_msg.lower():
                return "⚠️ O Gemini bloqueou a resposta por filtros de segurança. Tente reformular."
            return f"Erro técnico: {error_msg}"