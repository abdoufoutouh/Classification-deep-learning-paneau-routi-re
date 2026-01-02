import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Traffic Sign Dashboard",
    layout="wide"
)

# =========================
# LOAD MODEL AND CLASSES
# =========================
@st.cache_resource
def load_model_cached():
    try:
        return load_model("models/best_model.h5")
    except:
        st.error("❌ Modèle introuvable. Entraînez d'abord le modèle avec `python src/train.py`")
        return None

model = load_model_cached()

CLASS_NAMES = [
    'children',
    'no_entry',
    'pedestrian', 
    'road_work',
    'speed_30',
    'speed_50',
    'speed_70',
    'speed_80',
    'stop',
    'turn_left',
    'turn_right',
    'yield'
]

class_to_label = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# =========================
# UTILS FUNCTIONS
# =========================
def preprocess_image(image):
    """Prétraite une image pour MobileNetV2"""
    # Redimensionner en 224x224
    img = cv2.resize(image, (224, 224))
    # Convertir BGR en RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Preprocessing MobileNetV2
    img = preprocess_input(img)
    # Ajouter batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict_single_image(image):
    """Prédit une seule image"""
    if model is None:
        return None, 0.0
    
    processed_img = preprocess_image(image)
    preds = model.predict(processed_img, verbose=0)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))
    
    return CLASS_NAMES[class_id], confidence

def load_class_images(class_name, max_images=10):
    """Charge les images d'une classe spécifique"""
    class_path = f"data/processed/{class_name}"
    if not os.path.exists(class_path):
        return []
    
    images = []
    for img_name in os.listdir(class_path)[:max_images]:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    return images

def test_class_performance(class_name, max_images=20):
    """Teste les performances sur une classe spécifique"""
    if model is None:
        return 0.0, [], []
    
    images = load_class_images(class_name, max_images)
    if not images:
        return 0.0, [], []
    
    correct_predictions = []
    incorrect_predictions = []
    
    for img in images:
        pred_class, confidence = predict_single_image(img)
        
        if pred_class == class_name:
            correct_predictions.append((img, confidence))
        else:
            incorrect_predictions.append((img, pred_class, confidence))
    
    accuracy = len(correct_predictions) / len(images) if images else 0.0
    
    return accuracy, correct_predictions, incorrect_predictions

# =========================
# HEADER
# =========================
st.title("🚦 Traffic Sign Classification Dashboard")
st.markdown("**Système intelligent de reconnaissance des panneaux routiers avec Transfer Learning MobileNetV2**")

st.divider()

# =========================
# MODEL INFO
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("🧠 Modèle", "MobileNetV2 Transfer Learning")
col2.metric("🏷️ Classes", len(CLASS_NAMES))
col3.metric("📐 Taille image", "224 × 224")
col4.metric("⚙️ Preprocessing", "MobileNetV2")

st.divider()

# =========================
# MODE SELECTION
# =========================
st.subheader("🎯 Choisissez le mode de test")

mode = st.radio(
    "Sélectionnez un mode:",
    ["🖼️ Test image unique", "📊 Test par classe", "📈 Évaluation complète"],
    horizontal=True
)

st.divider()

# =========================
# MODE 1 — SINGLE IMAGE TEST
# =========================
if mode == "🖼️ Test image unique":
    st.subheader("🖼️ Test sur une image")
    
    col_upload, col_info = st.columns([1, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Téléverser une image de panneau routier",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            st.image(image, caption="Image d'entrée", use_column_width=True)
            
            if model is not None and st.button("🔍 Prédire", type="primary"):
                with st.spinner("Analyse en cours..."):
                    pred_class, confidence = predict_single_image(image)
                    
                    if pred_class:
                        st.success(f"**Panneau détecté : {pred_class.upper()}**")
                        st.info(f"**Confiance : {confidence * 100:.2f} %**")
                        
                        # Indicateur de confiance
                        if confidence > 0.85:
                            st.success("🟢 Prédiction très fiable")
                        elif confidence > 0.70:
                            st.warning("🟡 Prédiction fiable")
                        else:
                            st.error("🔴 Prédiction incertaine")
    
    with col_info:
        st.markdown("### ℹ️ Instructions")
        st.write("""
        1. Téléversez une image de panneau routier
        2. Cliquez sur **Prédire**
        3. Observez le résultat et la confiance
        
        **Formats acceptés :** JPG, JPEG, PNG
        **Taille recommandée :** 224×224 pixels
        """)

# =========================
# MODE 2 — CLASS TESTING
# =========================
elif mode == "📊 Test par classe":
    st.subheader("📊 Test par classe")
    
    col_select, col_results = st.columns([1, 2])
    
    with col_select:
        st.markdown("### 🎯 Sélectionnez une classe")
        selected_class = st.selectbox("Classe à tester:", CLASS_NAMES)
        
        max_images = st.slider("Nombre d'images à tester:", 5, 50, 20)
        
        if st.button("🧪 Lancer le test", type="primary"):
            if model is not None:
                with st.spinner(f"Test de la classe {selected_class}..."):
                    accuracy, correct, incorrect = test_class_performance(selected_class, max_images)
                    
                    # Stocker les résultats dans session state
                    st.session_state.class_test_results = {
                        'class': selected_class,
                        'accuracy': accuracy,
                        'correct': correct,
                        'incorrect': incorrect,
                        'total_tested': len(correct) + len(incorrect)
                    }
            else:
                st.error("❌ Modèle non chargé")
    
    # Afficher les résultats
    if 'class_test_results' in st.session_state:
        results = st.session_state.class_test_results
        
        with col_results:
            st.markdown(f"### 📊 Résultats pour {results['class'].upper()}")
            
            # Métriques principales
            col_acc, col_total, col_correct, col_incorrect = st.columns(4)
            
            col_acc.metric("🎯 Accuracy", f"{results['accuracy']*100:.1f}%")
            col_total.metric("📸 Testées", results['total_tested'])
            col_correct.metric("✅ Correctes", len(results['correct']))
            col_incorrect.metric("❌ Incorrectes", len(results['incorrect']))
            
            # Exemples corrects
            if results['correct']:
                st.markdown("#### ✅ Prédictions correctes")
                cols = st.columns(min(4, len(results['correct'])))
                for i, (img, conf) in enumerate(results['correct'][:4]):
                    with cols[i]:
                        st.image(img, caption=f"Confiance: {conf:.2f}", use_column_width=True)
            
            # Exemples incorrects
            if results['incorrect']:
                st.markdown("#### ❌ Prédictions incorrectes")
                cols = st.columns(min(4, len(results['incorrect'])))
                for i, (img, pred_class, conf) in enumerate(results['incorrect'][:4]):
                    with cols[i]:
                        st.image(img, caption=f"Prédit: {pred_class} ({conf:.2f})", use_column_width=True)

# =========================
# MODE 3 — COMPLETE EVALUATION
# =========================
elif mode == "📈 Évaluation complète":
    st.subheader("📈 Évaluation complète du modèle")
    
    if st.button("🚀 Lancer l'évaluation complète", type="primary"):
        if model is not None:
            with st.spinner("Évaluation en cours..."):
                # Charger toutes les données
                all_results = {}
                total_correct = 0
                total_images = 0
                
                progress_bar = st.progress(0)
                
                for i, class_name in enumerate(CLASS_NAMES):
                    accuracy, correct, incorrect = test_class_performance(class_name, 30)
                    all_results[class_name] = {
                        'accuracy': accuracy,
                        'correct': len(correct),
                        'incorrect': len(incorrect),
                        'total': len(correct) + len(incorrect)
                    }
                    
                    total_correct += len(correct)
                    total_images += len(correct) + len(incorrect)
                    
                    progress_bar.progress((i + 1) / len(CLASS_NAMES))
                
                # Accuracy globale
                global_accuracy = total_correct / total_images if total_images > 0 else 0.0
                
                st.success(f"🎯 **Accuracy globale : {global_accuracy*100:.2f}%** ({total_correct}/{total_images} images)")
                
                # Tableau des résultats par classe
                st.markdown("### 📊 Résultats détaillés par classe")
                
                results_data = []
                for class_name, metrics in all_results.items():
                    results_data.append({
                        'Classe': class_name.upper(),
                        'Accuracy': f"{metrics['accuracy']*100:.1f}%",
                        'Correctes': metrics['correct'],
                        'Incorrectes': metrics['incorrect'],
                        'Total': metrics['total']
                    })
                
                st.dataframe(results_data, use_container_width=True)
                
                # Graphique des accuracies par classe
                st.markdown("### 📈 Accuracy par classe")
                
                classes = list(all_results.keys())
                accuracies = [all_results[c]['accuracy'] for c in classes]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(classes, accuracies, color=['#1f77b4' if acc > 0.8 else '#ff7f0e' if acc > 0.6 else '#d62728' for acc in accuracies])
                ax.set_title('Accuracy par Classe')
                ax.set_xlabel('Classes')
                ax.set_ylabel('Accuracy')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                
                # Ajouter les valeurs sur les barres
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Analyse des classes problématiques
                st.markdown("### ⚠️ Classes nécessitant une attention")
                
                problematic_classes = [(c, m) for c, m in all_results.items() if m['accuracy'] < 0.7]
                
                if problematic_classes:
                    for class_name, metrics in problematic_classes:
                        st.warning(f"**{class_name.upper()}** : Accuracy {metrics['accuracy']*100:.1f}% ({metrics['correct']}/{metrics['total']})")
                else:
                    st.success("🎉 Toutes les classes ont une accuracy > 70% !")
        
        else:
            st.error("❌ Modèle non chargé")

# =========================
# FOOTER
# =========================
st.divider()
st.markdown("""
**🚦 Traffic Sign Classification System**  
*Powered by MobileNetV2 Transfer Learning*  
*Dataset: GTSRB - 12 classes*
""")

