import streamlit as st
import os
from dotenv import load_dotenv

# --- Imports LangChain (Assurez-vous qu'ils sont corrects) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import initialize_agent
import streamlit as st
import requests
import io # Pour gérer les données binaires des fichiers

# --- PARTIE A : Uploader et Classifier ---
load_dotenv()

GEO_SERVICE_API_KEY = os.getenv("GEO_SERVICE_API_KEY")
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL")
st.header("Give me the trash items you want to classify🗑️")

# Utilisation du widget d'upload Streamlit, acceptant plusieurs fichiers
uploaded_files = st.file_uploader(
    "Send the images you cant to classify:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
# Instanciation of final trash list but in a way not to be cleaned each time session is updated during chatbot discussion
if "final_trash_list" not in st.session_state:
    st.session_state.final_trash_list = []

def classify_images(files):
    """Appelle le service d'inférence fichier par fichier."""

    if not files:
        return

    st.info(f"Classification de {len(files)} image(s) en cours...")

    # 💡 L'AgentExecutor doit recevoir le résultat final

    with st.spinner("Analyse des images en cours..."):
        for file in files:
            # Préparation du dictionnaire 'files' pour UN SEUL fichier
            # Clé: le nom attendu par votre serveur (souvent 'file' ou 'image')
            files_to_send = {
                "file": (file.name, file.getvalue(), file.type)
            }

            try:
                # 1. Envoyer UN SEUL fichier à la fois
                response = requests.post(
                    f"{INFERENCE_SERVICE_URL}/predict",
                    files=files_to_send, # Envoi du dictionnaire de fichiers UNIQUE
                    timeout=30
                )
                print(f"Contenu brut de la réponse : {response.text}")
                response.raise_for_status() # Cette ligne lèvera l'erreur si le statut est 4xx/5xx
                result_list = response.json()

                # 2. On suppose que la réponse contient une liste de résultats
                # et que vous voulez la classe du premier (et seul) résultat
                if result_list and isinstance(result_list, list) and 'class' in result_list[0]:
                    st.session_state.final_trash_list.append(result_list[0]['class'])
                else:
                    st.warning(f"Réponse API inattendue pour {file.name}.")

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur HTTP ou de connexion pour {file.name}. Détail : {e}")
                break # Arrêter si une requête échoue
            except Exception as e:
                st.error(f"Erreur inattendue lors du traitement de {file.name}. Détail : {e}")
                break

        if st.session_state.final_trash_list:
            unique_classes = set(st.session_state.final_trash_list)

            # 2. Convertir l'ensemble de classes uniques en une chaîne de caractères
            classes_summary = ", ".join(unique_classes)

            # 3. Afficher le message de succès avec le résumé
            st.success(f"✅ Classification terminée ! Classes trouvées : {classes_summary}.")
            st.info("Vous pouvez maintenant demander à l'agent d'optimiser le trajet, par exemple : 'Minimise le temps de trajet en voiture pour déposer ces déchets'.")


# --- Logique d'appel Streamlit (reste la même) ---
if uploaded_files:
    # On vérifie si les fichiers ont changé pour ne pas reclasser inutilement
    current_file_names = {f.name for f in uploaded_files}
    if st.session_state.get('last_uploaded_files') != current_file_names:
        classify_images(uploaded_files)
        st.session_state.last_uploaded_files = current_file_names

# ... (Reste de l'affichage des résultats et de la boucle de chat)
st.markdown("---")

# --- PARTIE B : Boucle de Chat (Reste la même) ---
# ... (le code de st.title, agent initialization, chat_input, etc. suit ici)
st.title("🤖 Chatbot Analyse BigQuery (SQL Agent)")
# Initialisation de l'agent une seule fois
agent = initialize_agent()

# Gérer l'ID de session unique pour chaque utilisateur Streamlit
session_id = st.session_state.get("session_id", "default_user_1")
config = {"configurable": {"session_id": session_id}}

# Afficher l'historique de chat existant
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Gestion de la nouvelle entrée utilisateur
if prompt := st.chat_input("Posez une question sur les points de collecte..."):

    # 1. Afficher la question de l'utilisateur (votre code existant)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- NOUVEAU : Préparation du Prompt Enrichi ---
    full_prompt = prompt

    # Vérifier si nous avons des classes de déchets en mémoire
    if st.session_state.get('final_trash_list'):

        # 1. Obtenir la liste des classes uniques
        unique_classes = set(st.session_state.final_trash_list)
        classes_summary = ", ".join(unique_classes)

        # 2. Enrichir le prompt de l'utilisateur avec le contexte des déchets
        # Ceci est la technique d'Injection de Contexte
        full_prompt = (
            f"CONTEXTE DÉCHETS: Les types de déchets que je dois déposer sont : {classes_summary}. "
            f"QUESTION UTILISATEUR: {prompt}"
        )

  # ... (Dans la boucle de chat_input, au niveau de l'étape 2. Exécuter l'agent)

    # 2. Exécuter l'agent
    with st.spinner("L'agent SQL analyse la base de données..."):
        try:
            response = agent.invoke(
                {"input": full_prompt},
                config=config
            )

            raw_output = response["output"]
            agent_response = "" # Initialisation de la réponse finale

            # --- DÉBUT CORRECTION ROBUSTE ---

            if isinstance(raw_output, list):
                # Si c'est une liste, nous cherchons le texte dans CHAQUE élément

                all_text_parts = []
                for item in raw_output:
                    if isinstance(item, dict) and 'text' in item:
                        # Cas 1: C'est le format {'type': 'text', 'text': '...'}. On prend le texte.
                        all_text_parts.append(item['text'])
                    elif isinstance(item, str):
                        # Cas 2: C'est la suite de la chaîne qui a été coupée (votre cas récent).
                        all_text_parts.append(item)
                    # Si c'est un autre type (comme 'tool_call', on l'ignore ici)

                # Joindre toutes les parties de texte en une seule réponse
                agent_response = "\n".join(all_text_parts)

            else:
                # Si c'est déjà une chaîne de caractères simple (le cas idéal)
                agent_response = str(raw_output)

            # --- FIN CORRECTION ROBUSTE ---

        except Exception as e:
            agent_response = f"Désolé, une erreur est survenue lors de l'exécution : {e}"
            st.error(agent_response)

    # 3. Afficher la réponse de l'agent
    with st.chat_message("assistant"):
        # Utiliser st.markdown pour bien formater la liste (bullets, retours à la ligne)
        st.markdown(agent_response)

    # 4. Enregistrer la réponse
    st.session_state.messages.append({"role": "assistant", "content": agent_response})


    # # --- 5) Exécuter des Requêtes Conversationnelles ---

    # # Définition de l'ID de session
    # session_id = "user_session_123"
    # config = {"configurable": {"session_id": session_id}}

    # # 💡 CORRECTION 1 : Appeler l'agent conversationnel (le wrapper)
    # # 💡 CORRECTION 2 : Utiliser la clé "input" avec une chaîne de caractères
    # response = conversational_agent.invoke(
    #     {"input": "How many places can take batteries?"},
    #     config=config
    # )

    # # L'output est maintenant directement le contenu textuel de la réponse finale
    # pprint(response["output"])

    # print("\n--- Interaction 2 (Test de la mémoire) ---")
    # # Une seconde requête pour tester la mémoire
    # response_2 = conversational_agent.invoke(
    #     {"input": "Which one is the closest to Nantes city center?"},
    #     config=config
    # )
    # pprint(response_2["output"])
