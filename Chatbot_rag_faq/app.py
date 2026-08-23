import streamlit as st  # Bibliothèque pour créer l'interface web
from rag_engine import extract_chunks_from_pdf, SimpleRAG  # Importation du moteur RAG

# Configuration générale de la page Streamlit (Titre de l'onglet et disposition)
st.set_page_config(page_title="Chatbot FAQ RAG", page_icon="🤖", layout="wide")

# Titre principal et description de l'application
st.title("🤖 Chatbot FAQ RAG — Assistance Documentaire")
st.write("Posez vos questions sur le document d'étude .")


# Fonction d'initialisation du moteur RAG mise en cache (exécutée une seule fois au démarrage)
@st.cache_resource
def init_rag():
    # Découpage du fichier PDF situé dans le dossier 'data/'
    chunks = extract_chunks_from_pdf("data/document.pdf")
    # Création de l'index RAG TF-IDF
    return SimpleRAG(chunks)


# Vérification de la présence du fichier PDF avant de lancer l'application
try:
    rag = init_rag()
except Exception as e:
    st.error(" Fichier introuvable : Assurez-vous d'avoir placé le fichier 'document.pdf' dans le dossier 'data/'.")
    st.stop()

# Initialisation de l'historique de discussion dans la session utilisateur si inexistant
if "messages" not in st.session_state:
    st.session_state.messages = []

# Réaffichage de tous les messages précédents stockés dans l'historique de la session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Affichage du bloc source si le message provient de l'assistant
        if "source" in msg:
            st.info(f" **Source :** Page {msg['source']['page']} | Score de confiance : {msg['source']['confidence']*100:.1f}%")

# Zone de saisie d'un nouveau message par l'utilisateur
if prompt := st.chat_input("Posez votre question sur le document..."):
    
    # 1. Enregistrement et affichage de la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Recherche de la réponse via le moteur RAG
    res = rag.query(prompt)
    
    # Formatage de la réponse de l'assistant sous forme de citation
    response_text = f"D'après le document :\n\n> *\"{res['answer']}\"*"

    # 3. Affichage de la réponse de l'assistant avec citation et score de confiance
    with st.chat_message("assistant"):
        st.markdown(response_text)
        st.info(f" **Source :** Page {res['page']} | Score de confiance : {res['confidence']*100:.1f}%")

    # 4. Enregistrement de la réponse de l'assistant dans l'historique de session
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "source": {"page": res["page"], "confidence": res["confidence"]}
    })