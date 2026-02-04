import streamlit as st
import os
import glob
from src.data_ingestion import DataIngestion
from src.agent_core import DataAnalysisAgent

# --- Configuração da Página ---
st.set_page_config(
    page_title="Data Agent AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente de Análise de Dados")

# --- Barra Lateral (Configuração) ---
with st.sidebar:
    st.header("📂 Configuração")
    
    # Lista arquivos na pasta 'data'
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        
    files = [f for f in os.listdir(data_folder) if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    selected_file = st.selectbox("Escolha a base de dados:", ["Selecione..."] + files)
    
    # Botão de Reset
    if st.button("Limpar Conversa"):
        st.session_state.messages = []
        st.session_state.agent = None
        st.rerun()

# --- Gerenciamento de Estado (Memória da Sessão) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

# --- Lógica de Carregamento do Agente ---
if selected_file != "Selecione..." and selected_file is not None:
    file_path = os.path.join(data_folder, selected_file)
    
    # Só recarrega se o agente ainda não existir ou se mudou o arquivo
    if st.session_state.agent is None:
        with st.spinner('Carregando e indexando dados...'):
            try:
                df = DataIngestion.load_data(file_path)
                # Instancia o agente
                st.session_state.agent = DataAnalysisAgent(df)
                st.success(f"Base '{selected_file}' carregada com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar arquivo: {e}")

# --- Interface de Chat ---
# 1. Exibe histórico anterior
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Se houver imagem associada a essa mensagem, exibe
        if "image" in message:
            st.image(message["image"])

# 2. Input do Usuário
if prompt := st.chat_input("Pergunte sobre seus dados..."):
    
    if st.session_state.agent is None:
        st.warning("Por favor, selecione um arquivo na barra lateral primeiro.")
    else:
        # Exibe mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)
        # Salva no histórico
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Processamento do Agente
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                response = st.session_state.agent.process_message(prompt)
                st.markdown(response)
                
                # Verifica se um gráfico foi gerado recentemente
                # Lógica: O agente diz "Gráfico salvo em: outputs/nome.png"
                image_path = None
                if "outputs/" in response and ".png" in response:
                    try:
                        # Extrai o caminho (procura a string outputs/...)
                        words = response.split()
                        for word in words:
                            if "outputs/" in word and ".png" in word:
                                # Limpa pontuação final se houver (ex: .png.)
                                clean_path = word.rstrip(".,;:")
                                if os.path.exists(clean_path):
                                    st.image(clean_path, caption="Gráfico Gerado")
                                    image_path = clean_path
                    except:
                        pass # Falha silenciosa na extração da imagem
                
                # Salva resposta no histórico
                msg_data = {"role": "assistant", "content": response}
                if image_path:
                    msg_data["image"] = image_path
                
                st.session_state.messages.append(msg_data)