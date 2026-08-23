# Chatbot RAG / FAQ sur Document PDF (Sujet E)

Ce projet met en œuvre un système de **Retrieval-Augmented Generation (RAG)** permettant de poser des questions 
en langage naturel sur un document PDF volumineux (+100 pages) et d'obtenir des réponses basées sur le contexte extrait.

---

##  Structure du Projet

```text
Chatbot_rag_faq/
├── data/
│   ├── document.pdf          # Document source (+100 pages)
│   └── test_set_20.json      # Dataset d'évaluation (20 questions/réponses)
├── rag_engine.py             # Moteur d'indexation (TF-IDF) et de recherche contextuelle
├── app.py                    # Interface utilisateur Streamlit
├── evaluate.py               # Script de calcul des métriques (Exact Match & F1-Score)
└── README.md                 # Documentation du projet
```

---

##  Configuration et Installation

### 1. Prérequis
- Python 3.10 ou supérieur

### 2. Création et activation de l'environnement virtuel
```bash
# Création de l'environnement venv
python -m venv venv

# Activation sous Windows (Git Bash)
source venv/Scripts/activate
```

### 3. Installation des dépendances
```bash
pip install pymupdf scikit-learn streamlit
```

---

##  Utilisation

### 1. Démarrer l'application Web (Streamlit)
```bash
streamlit run app.py
```
L'interface s'ouvre automatiquement sur `http://localhost:8501`.

### 2. Lancer l'évaluation automatique
Pour évaluer le système sur le jeu de données `test_set_20.json` :
```bash
python evaluate.py
```

---

##  Métriques d'Évaluation
- **Exact Match (EM) :** Mesure la correspondance exacte entre la réponse prédite et la réponse attendue.
- **F1-Score :** Mesure le chevauchement (overlap) au niveau des mots entre la réponse générée et la réponse de référence.
