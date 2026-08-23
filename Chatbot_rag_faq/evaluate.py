import json  # Lecture du fichier de test au format JSON
import re  # Nettoyage des chaînes de caractères
from collections import Counter  # Calcul d'intersection de mots pour le F1-score
from rag_engine import extract_chunks_from_pdf, SimpleRAG  # Importation du moteur RAG


def normalize_text(text):
    """Standardise le texte : passage en minuscules, suppression de la ponctuation et des articles."""
    text = text.lower()  # Passage en minuscules
    text = re.sub(r'\b(a|an|the|le|la|les|un|une|des)\b', ' ', text)  # Mots vides courants
    text = re.sub(r'[^\w\s]', '', text)  # Suppression de la ponctuation
    return " ".join(text.split())  # Suppression des espaces superflus


def compute_exact_match(prediction, truth):
    """Calcule si la prédiction correspond exactement à la réponse attendue (1 ou 0)."""
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    """Calcule le F1-Score au niveau des mots entre la réponse prédite et la réponse attendue."""
    pred_tokens = normalize_text(prediction).split()  # Mots de la prédiction
    truth_tokens = normalize_text(truth).split()      # Mots de la réponse attendue

    # Compte les mots communs entre la prédiction et la vérité terrain
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    # Cas limites : si l'une des listes est vide
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0.0

    # Calcul de la Précision et du Rappel
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    
    # Calcul de la moyenne harmonique (F1-Score)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def run_evaluation():
    """Exécute l'évaluation complète sur l'ensemble de test des 20 questions-réponses."""
    print(" Initialisation du moteur RAG et chargement du document...")
    chunks = extract_chunks_from_pdf("data/document.pdf")
    rag = SimpleRAG(chunks)

    # Chargement du fichier JSON contenant le jeu de test
    with open("data/test_set_20.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)

    em_scores = []  # Liste des scores Exact Match
    f1_scores = []  # Liste des scores F1

    print("\n Lancement de l'évaluation sur 20 questions...\n")

    # Parcours des 20 questions du dataset de test
    for idx, item in enumerate(test_set, 1):
        res = rag.query(item["question"])
        pred = res["answer"]
        truth = item["expected_answer"]

        # Calcul des scores pour la question courante
        em = compute_exact_match(pred, truth)
        f1 = compute_f1(pred, truth)

        em_scores.append(em)
        f1_scores.append(f1)

        print(f"Q{idx}: {item['question']}")
        print(f" -> Exact Match : {em} | F1-Score : {f1:.2f}\n")

    # Affichage des résultats globaux
    print("==========================================")
    print("📊 RÉSULTATS D'ÉVALUATION DU RAG")
    print("==========================================")
    print(f"Nombre de questions évaluées : {len(test_set)}")
    print(f"Exact Match (EM) Moyenne     : {sum(em_scores)/len(em_scores)*100:.2f}%")
    print(f"F1-Score Moyen               : {sum(f1_scores)/len(f1_scores)*100:.2f}%")
    print("==========================================")


if __name__ == "__main__":
    run_evaluation()