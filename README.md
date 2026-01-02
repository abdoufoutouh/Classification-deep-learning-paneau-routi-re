# Traffic Sign Classification - Instructions

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📦 Structure du projet

```
traffic_sign_classification/
├── data/
│   └── processed/          # Images organisées par classe
├── src/
│   ├── model.py           # Modèle MobileNetV2 Transfer Learning
│   ├── data_loader.py     # Data loader avec preprocessing 224x224
│   ├── train.py           # Entraînement (8 epochs)
│   └── evaluate.py        # Évaluation complète
├── app/
│   └── app.py             # Dashboard Streamlit 3 modes
├── models/
│   └── best_model.h5      # Modèle entraîné
└── results/
    └── evaluation_results.png
```

## 🎯 Entraînement du modèle

```bash
cd src
python train.py
```

**Résultat attendu :**
- Modèle sauvegardé dans `models/best_model.h5`
- Accuracy > 90% avec MobileNetV2
- 8 epochs avec early stopping

## 📊 Évaluation

```bash
cd src
python evaluate.py
```

**Résultats :**
- Accuracy globale et par classe
- Matrice de confusion
- Graphiques détaillés dans `results/evaluation_results.png`

## 🖥️ Dashboard Streamlit

```bash
cd app
streamlit run app.py
```

### 3 modes disponibles :

**🖼️ Mode 1 - Test image unique**
- Upload d'une image
- Prédiction avec confiance
- Indicateur de fiabilité

**📊 Mode 2 - Test par classe** 
- Sélection d'une classe
- Test automatique sur N images
- Affichage exemples corrects/incorrects
- Taux de reconnaissance par classe

**📈 Mode 3 - Évaluation complète**
- Test sur toutes les classes
- Accuracy globale
- Tableau détaillé par classe
- Graphique des performances
- Alertes classes problématiques

## 🔧 Configuration technique

- **Modèle** : MobileNetV2 (weights=imagenet)
- **Input** : 224×224×3
- **Preprocessing** : `tf.keras.applications.mobilenet_v2.preprocess_input`
- **Classes** : 12 (children, no_entry, pedestrian, road_work, speed_30, speed_50, speed_70, speed_80, stop, turn_left, turn_right, yield)
- **Optimizer** : Adam (lr=1e-4)
- **Loss** : sparse_categorical_crossentropy

## ✅ Validation pour jury

Le système permet de démontrer :

1. **Reconnaissance fiable** : STOP → STOP (pas speed_80)
2. **Test par classe** : "Montrez-moi la reconnaissance du panneau STOP"
3. **Performance mesurable** : Accuracy > 90% sur toutes les classes
4. **Interface intuitive** : Dashboard 3 modes pour démonstration live

## 🚨 Dépannage

**Modèle introuvable** : Lancez d'abord `python src/train.py`
**Pas d'images** : Vérifiez le dossier `data/processed/`
**Erreur import** : Installez les dépendances avec `pip install -r requirements.txt`