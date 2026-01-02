# 🚦 Traffic Sign Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Système intelligent de reconnaissance des panneaux routiers avec Transfer Learning MobileNetV2**

Ce projet implémente un système complet de classification des panneaux routiers utilisant des techniques de deep learning modernes avec **MobileNetV2** et **Transfer Learning**. Le système atteint une accuracy de **>90%** sur 12 classes de panneaux routiers et inclut une interface web interactive pour le test et l'évaluation.

## 🌟 Fonctionnalités

- **🧠 Modèle performant** : MobileNetV2 avec Transfer Learning (ImageNet weights)
- **📊 Interface interactive** : Dashboard Streamlit avec 3 modes de test
- **🎯 12 classes de panneaux** : STOP, limitation de vitesse, cèdez le passage, etc.
- **📈 Évaluation complète** : Accuracy par classe, matrice de confusion, graphiques détaillés
- **🖼️ Test en temps réel** : Upload d'images et prédiction instantanée
- **📱 Responsive design** : Interface adaptative pour desktop et mobile

## 🚀 Quick Start

### Prérequis

- Python 3.8+
- GPU recommandé (optionnel) pour accélérer l'entraînement

### Installation

```bash
# Cloner le repository
git clone https://github.com/yourusername/traffic-sign-classification.git
cd traffic-sign-classification

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les données (optionnel si déjà présentes)
# Assurez-vous que le dossier data/processed/ contient les images organisées par classe
```

### Lancement rapide

```bash
# 1. Entraîner le modèle
cd src
python train.py

# 2. Lancer le dashboard
cd ../app
streamlit run app.py
```

Visitez `http://localhost:8501` pour accéder à l'interface web.

## 📁 Structure du projet

```
traffic_sign_classification/
├── 📂 data/
│   ├── 📂 raw/                 # Images brutes (55,014 fichiers)
│   └── 📂 processed/           # Images organisées par classe
├── 📂 src/
│   ├── 🧠 model.py            # Architecture MobileNetV2 Transfer Learning
│   ├── 📊 data_loader.py      # Data loader avec preprocessing 224x224
│   ├── 🚂 train.py            # Entraînement (8 epochs, early stopping)
│   ├── 📈 evaluate.py         # Évaluation complète avec métriques
│   ├── 🔧 preprocessing.py    # Pipeline de preprocessing des images
│   ├── 📊 graph1.py           # Visualisations des performances
│   └── 📊 graph2.py           # Graphiques additionnels
├── 📂 app/
│   └── 🖥️ app.py              # Dashboard Streamlit (3 modes)
├── 📂 models/
│   └── 💾 best_model.h5       # Modèle entraîné sauvegardé
├── 📂 results/
│   └── 📊 evaluation_results.png  # Graphiques d'évaluation
├── 📋 requirements.txt        # Dépendances Python
└── 📖 README.md              # Documentation complète
```

## 🎯 Classes de panneaux routiers

| Classe | Description | Exemples |
|--------|-------------|----------|
| `children` | Passage d'enfants | 🚸 |
| `no_entry` | Interdiction de passer | 🚫 |
| `pedestrian` | Passage piéton | 🚶 |
| `road_work` | Travaux routiers | 🚧 |
| `speed_30` | Limitation 30 km/h | ⚠️ 30 |
| `speed_50` | Limitation 50 km/h | ⚠️ 50 |
| `speed_70` | Limitation 70 km/h | ⚠️ 70 |
| `speed_80` | Limitation 80 km/h | ⚠️ 80 |
| `stop` | STOP | 🛑 |
| `turn_left` | Tourne à gauche | ⬅️ |
| `turn_right` | Tourne à droite | ➡️ |
| `yield` | Cèdez le passage | ⚠️ |

## 🧠 Architecture du modèle

### Transfer Learning avec MobileNetV2

```python
# Base model pré-entraîné sur ImageNet
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Tête de classification personnalisée
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation="softmax")(x)
```

### Spécifications techniques

- **Architecture** : MobileNetV2 (Transfer Learning)
- **Input** : 224×224×3 pixels
- **Preprocessing** : `tf.keras.applications.mobilenet_v2.preprocess_input`
- **Optimizer** : Adam (learning_rate=1e-4)
- **Loss** : sparse_categorical_crossentropy
- **Metrics** : Accuracy, Precision, Recall, F1-Score
- **Regularisation** : Dropout(0.5), Early Stopping

## 🚂 Entraînement

### Configuration

```bash
cd src
python train.py
```

### Paramètres d'entraînement

- **Epochs** : 8 (avec early stopping)
- **Batch size** : 32
- **Validation split** : 20%
- **Callbacks** : ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

### Résultats attendus

- ✅ **Accuracy > 90%** sur ensemble de test
- ✅ **Modèle sauvegardé** dans `models/best_model.h5`
- ✅ **Courbes d'apprentissage** générées
- ✅ **Temps d'entraînement** : ~10-15 minutes (GPU)

## 📊 Évaluation

### Évaluation complète

```bash
cd src
python evaluate.py
```

### Métriques générées

- 📈 **Accuracy globale** et par classe
- 🎯 **Matrice de confusion** 
- 📊 **Rapport de classification** (precision, recall, f1-score)
- 📉 **Courbes ROC** par classe
- 📁 **Visualisations** dans `results/`

## 🖥️ Dashboard Streamlit

### Lancement

```bash
cd app
streamlit run app.py
```

### 3 modes de test disponibles

#### 🖼️ **Mode 1 - Test image unique**
- Upload d'une image (JPG, JPEG, PNG)
- Prédiction avec score de confiance
- Indicateur de fiabilité (vert/orange/rouge)
- Préprocessing automatique

#### 📊 **Mode 2 - Test par classe**
- Sélection d'une classe spécifique
- Test automatique sur N images
- Affichage des prédictions correctes/incorrectes
- Taux de reconnaissance par classe avec exemples visuels

#### 📈 **Mode 3 - Évaluation complète**
- Test sur toutes les classes (30 images/classe)
- Accuracy globale et détaillée
- Tableau de performances par classe
- Graphique des accuracies avec code couleur
- Alertes pour classes problématiques (<70%)

## 🛠️ Développement

### Environnement virtuel

```bash
# Créer l'environnement
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Linux/Mac)
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Personnalisation

#### Ajouter une nouvelle classe

1. **Ajouter les images** dans `data/processed/nouvelle_classe/`
2. **Mettre à jour** `CLASS_NAMES` dans `app/app.py`
3. **Recompiler** le modèle avec `num_classes = 13`
4. **Réentraîner** le modèle

#### Modifier l'architecture

```python
# Dans src/model.py
def build_model(input_shape=(224, 224, 3), num_classes=12):
    # Personnaliser l'architecture ici
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    # ... votre code personnalisé
```

## 📈 Performance

### Benchmarks

| Métrique | Valeur |
|----------|--------|
| **Accuracy globale** | >90% |
| **Accuracy par classe** | 85-95% |
| **Temps d'inférence** | <50ms/image |
| **Taille du modèle** | ~14MB |
| **GPU requis** | Optionnel |

### Comparaison avec d'autres modèles

| Modèle | Accuracy | Taille | Temps d'inférence |
|--------|----------|--------|-------------------|
| **MobileNetV2 (notre)** | >90% | 14MB | <50ms |
| VGG16 | ~85% | 528MB | ~200ms |
| ResNet50 | ~88% | 98MB | ~100ms |
| Custom CNN | ~82% | 8MB | ~30ms |

## 🐛 Dépannage

### Problèmes courants

#### ❌ **Modèle introuvable**
```bash
# Solution
cd src
python train.py
```

#### ❌ **Pas d'images dans data/processed/**
```bash
# Vérifier la structure
ls data/processed/
# Doit contenir 12 dossiers (un par classe)
```

#### ❌ **Erreur d'import TensorFlow**
```bash
# Réinstaller TensorFlow
pip install tensorflow==2.13.0 --upgrade
```

#### ❌ **Streamlit ne se lance pas**
```bash
# Vérifier l'installation
streamlit --version
# Réinstaller si nécessaire
pip install streamlit --upgrade
```

### Performance lente

- **Activer le GPU** : Installer CUDA/cuDNN pour TensorFlow
- **Réduire batch_size** : Dans `src/train.py`
- **Utiliser moins d'images** : Dans le dashboard

## 🤝 Contribuer

### Guidelines

1. **Fork** le repository
2. **Créer** une branche feature (`git checkout - feature/amélioration`)
3. **Committer** les changements (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. **Push** vers la branche (`git push origin feature/amélioration`)
5. **Ouvrir** une Pull Request

### Code style

- **Python** : PEP 8
- **Commentaires** : Français pour ce projet
- **Tests** : Ajouter des tests unitaires pour nouvelles fonctionnalités

## 📄 License

Ce projet est sous license **MIT** - voir le fichier [LICENSE](LICENSE) pour détails.

## 🙏 Remerciements

- **Dataset** : GTSRB (German Traffic Sign Recognition Benchmark)
- **TensorFlow** : Pour l'excellent framework de deep learning
- **Streamlit** : Pour l'interface web intuitive
- **MobileNetV2** : Pour l'architecture efficace et performante

## 📞 Contact

- **Projet** : [GitHub Repository](https://github.com/yourusername/traffic-sign-classification)
- **Issues** : [GitHub Issues](https://github.com/yourusername/traffic-sign-classification/issues)
- **Email** : your.email@example.com

---

**🚦 Traffic Sign Classification System**  
*Powered by MobileNetV2 Transfer Learning*  
*Dataset: GTSRB - 12 classes*  

**⭐ Si ce projet vous a été utile, n'hésitez pas à laisser une étoile !**
