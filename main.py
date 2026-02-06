import sys
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente antes de qualquer coisa
load_dotenv()

# Tenta importar o agente com tratamento de erro amigável
try:
    from src.agent_core import DataAnalysisAgent
except ImportError as e:
    print("\n[ERRO CRÍTICO DE IMPORTAÇÃO]")
    print(f"Detalhes: {e}")
    print("DICA: Verifique se o seu ambiente virtual (.venv) está ativo e as bibliotecas atualizadas.")
    print("Tente rodar: pip install -U langchain langchain-community langchain-experimental")
    sys.exit(1)

def main():
    print("🤖 --- Agente Autônomo de Dados (Gemini 2.0 Flash Lite) ---")
    print(f"[SYSTEM] Inicializando e indexando arquivos na pasta 'data/'...")
    
    try:
        # O construtor do agente já chama o DatasetManager e indexa tudo
        agent = DataAnalysisAgent()
        print("[SYSTEM] Agente pronto! Pode perguntar sobre qualquer arquivo da pasta.")
        print("[INFO] Comandos: 'sair' para encerrar.\n")
    except Exception as e:
        print(f"[FATAL ERROR] Não foi possível iniciar o agente: {e}")
        return

    while True:
        try:
            user_input = input("\nVocê: ").strip()
        except KeyboardInterrupt:
            break
            
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("👋 Até logo!")
            break

        if not user_input:
            continue

        print("⏳ Pensando...")
        
        response = agent.process_message(user_input)
        
        print(f"\n🤖 Agente: {response}")

if __name__ == "__main__":
    main()