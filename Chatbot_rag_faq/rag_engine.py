import fitz  # PyMuPDF : Bibliothèque permettant d'extraire le texte d'un fichier PDF
import re  # Module de regex pour le nettoyage du texte (suppression des espaces superflus)
import numpy as np  # Calcul numérique
from sklearn.feature_extraction.text import TfidfVectorizer  # Vectorisation du texte (TF-IDF)
from sklearn.metrics.pairwise import cosine_similarity  # Mesure de similitude entre la question et les textes


def extract_chunks_from_pdf(pdf_path, chunk_size=300):
    """Extrait le texte d'un PDF long et le découpe en blocs (chunks) de mots.
    
    Chaque bloc conserve le numéro de la page d'où il provient.
    """
    # Ouverture du document PDF
    doc = fitz.open(pdf_path)
    chunks = []  # Liste pour stocker tous les blocs de texte extraits

    # Parcours de chaque page du PDF
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()  # Extraction du texte brut de la page
        
        # Nettoyage : remplace les sauts de ligne et espaces multiples par un seul espace
        text = re.sub(r'\s+', ' ', text).strip()

        # Découpage du texte en mots
        words = text.split()

        # Découpage du texte par blocs de 'chunk_size' mots (par défaut 300 mots)
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            
            # On conserve uniquement les blocs significatifs (plus de 50 caractères)
            if len(chunk_text) > 50:
                chunks.append({
                    "page": page_num + 1,  # Numéro de page (indexé à partir de 1)
                    "text": chunk_text     # Contenu du bloc de texte
                })
                
    return chunks


class SimpleRAG:
    """Classe qui gère l'indexation TF-IDF et la recherche du passage source le plus pertinent."""

    def __init__(self, chunks):
        self.chunks = chunks  # Stockage des blocs avec métadonnées
        self.texts = [c["text"] for c in chunks]  # Extraction uniquement des textes pour le TF-IDF
        
        # Initialisation du vectoriseur TF-IDF (mots simples + paires de mots, sans mots vides anglais)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        
        # Transforme l'ensemble des blocs de texte en une matrice numérique TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def query(self, question, top_k=1):
        """Recherche le passage le plus pertinent répondant à la question posée."""
        # Vectorisation de la question posée par l'utilisateur
        q_vec = self.vectorizer.transform([question])
        
        # Calcul de la similitude cosinus entre la question et chaque bloc de texte
        similarities = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        # Récupération de l'index du bloc ayant le score de similitude le plus élevé
        best_idx = np.argmax(similarities)
        
        # Score de confiance compris entre 0 et 1 (similitude cosinus)
        confidence = float(similarities[best_idx])

        # Récupération du bloc correspondant
        best_chunk = self.chunks[best_idx]
        
        return {
            "answer": best_chunk["text"],    # Passage source extrait du PDF
            "page": best_chunk["page"],        # Page source dans le document
            "confidence": confidence           # Score de confiance du résultat
        }