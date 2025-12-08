import os
from pprint import pprint
from IPython.display import Markdown
from dotenv import load_dotenv
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
# --- Imports pour la Mémoire (Corrigés) ---
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage # Non utilisé directement dans l'entrée, mais toujours utile
import streamlit as st
# --- Configuration Éphémère de l'Historique ---
# In the future I should improve this part to get the sesion ID to be linked with a user cookie

store: dict[str, BaseChatMessageHistory] = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Récupère ou crée l'historique de la session pour un ID donné."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
# --- FIN Configuration Mémoire ---
# --- 1) LLM Gemini 2.0 Flash ---

@st.cache_resource
def initialize_agent():
    model =  ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # --- 2) Connexion BigQuery ---
    # To get environment keys
    load_dotenv()
    GCP_PROJECT = os.environ.get("GCP_PROJECT")
    GCP_DATASET = os.environ.get("GCP_DATASET")
    db = SQLDatabase.from_uri(
        f"bigquery://{GCP_PROJECT}?dataset={GCP_DATASET}",
        # Assurez-vous que GCP_DATASET est la variable contenant le nom du dataset
        include_tables=[f"{GCP_DATASET}.trash_collection_points"]
    )


    # --- 3) Création des tools SQL ---
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    custom_prompt = (
        "Tu es un assistant expert pour la base de données BigQuery contenant les points de collecte de déchets. "
        "Ton rôle est de répondre aux questions de l'utilisateur en français ou en anglais en utilisant UNIQUEMENT les informations de la base de données. "
        "Ne génère jamais de code SQL si l'utilisateur pose une question de conversation générale (ex: 'bonjour'). "
        "Lorsque tu trouves une information, formule la réponse de manière aimable, complète et concise, sans afficher le code SQL brut."
        "L'utilisateur peut te donner une adresse à partir de laquelle il souhaite déposer les modéles - si il n'en donne pas par défaut c'est le centre ville de Nantes"
        "Si l'utilisateur a besoin de déposer des déchets à plusieurs endroits propose lui soit l'option d'un endroit unique où il peut tout déposer ou une liste des points les plus près de l'adresse donnée"
        "Si l'utilisateur mentionne du textile alors il faut correspondre à textile trash"
    )
    # --- 4) Créer un AgentExecutor simple ---
    agent_executor = create_sql_agent(
        llm=model,
        db=db,
        agent_type="openai-tools", # Un type d'agent performant pour les modèles Gemini/OpenAI
        extra_prompt_instructions=custom_prompt,
        verbose=False )

    # --- 4) Rendre l'Agent Conversational ---
    conversational_agent = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input", # Clé d'entrée attendue par l'AgentExecutor
        history_messages_key="chat_history", # Clé où stocker l'historique
    )
    return conversational_agent
