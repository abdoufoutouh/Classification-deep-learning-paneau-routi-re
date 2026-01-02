import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
from data_loader import X, y, CLASS_NAMES

def evaluate_model():
    """Évaluation complète du modèle avec toutes les métriques"""
    
    # Vérifier que le modèle existe
    try:
        model = load_model("models/best_model.h5")
        print("✅ Modèle chargé avec succès")
    except:
        print("❌ Erreur: Modèle introuvable. Entraînez d'abord le modèle avec src/train.py")
        return
    
    # Vérifier que les données sont chargées
    if X.shape[0] == 0:
        print("❌ Erreur: Aucune donnée chargée")
        return
    
    print(f"\n📊 Évaluation sur {X.shape[0]} images")
    print(f"🏷️  {len(CLASS_NAMES)} classes: {CLASS_NAMES}")
    
    # Prédictions
    print("\n🔮 Prédictions en cours...")
    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Métriques globales
    accuracy = accuracy_score(y, y_pred)
    print(f"\n🎯 Accuracy globale: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Rapport de classification détaillé
    print("\n📋 Rapport de classification:")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))
    
    # Matrice de confusion
    print("\n📈 Génération de la matrice de confusion...")
    cm = confusion_matrix(y, y_pred)
    
    # Visualisation
    plt.figure(figsize=(12, 10))
    
    # Matrice de confusion
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matrice de Confusion')
    plt.xlabel('Classe Prédite')
    plt.ylabel('Classe Réelle')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Accuracy par classe
    plt.subplot(2, 2, 2)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    plt.bar(CLASS_NAMES, class_accuracies)
    plt.title('Accuracy par Classe')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Distribution des prédictions
    plt.subplot(2, 2, 3)
    plt.hist(y_pred, bins=len(CLASS_NAMES), alpha=0.7, label='Prédictions')
    plt.hist(y, bins=len(CLASS_NAMES), alpha=0.7, label='Réel')
    plt.title('Distribution des Classes')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'images')
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.legend()
    
    # Confidence scores
    plt.subplot(2, 2, 4)
    confidence_scores = np.max(y_pred_proba, axis=1)
    plt.hist(confidence_scores, bins=20, alpha=0.7)
    plt.title('Distribution des Confiances')
    plt.xlabel('Score de Confiance')
    plt.ylabel('Nombre d\'images')
    
    plt.tight_layout()
    plt.savefig('results/evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyse par classe
    print("\n🔍 Analyse détaillée par classe:")
    print("-" * 60)
    for i, class_name in enumerate(CLASS_NAMES):
        class_indices = np.where(y == i)[0]
        if len(class_indices) > 0:
            class_correct = np.sum(y_pred[class_indices] == i)
            class_total = len(class_indices)
            class_acc = class_correct / class_total
            avg_confidence = np.mean(np.max(y_pred_proba[class_indices], axis=1))
            print(f"{class_name:12} : {class_acc:.3f} ({class_correct}/{class_total}) - Confiance: {avg_confidence:.3f}")
    
    print(f"\n✅ Évaluation terminée. Résultats sauvegardés dans results/evaluation_results.png")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    evaluate_model()
