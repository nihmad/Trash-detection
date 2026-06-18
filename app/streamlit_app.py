"""Application Streamlit de démonstration.

Permet de charger une image, de sélectionner un modèle entraîné, puis
d'afficher la classe prédite et le score de confiance.

Lancement :

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Rend le package ``src`` importable lorsque Streamlit exécute ce fichier.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import resize_image  # noqa: E402
from src.utils import CLASS_NAMES, IMAGE_SIZE, MODELS_DIR  # noqa: E402


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_trained_model(model_name: str):
    """Charge un modèle Keras depuis le répertoire ``models/``.

    Le décorateur de cache évite de recharger le modèle à chaque interaction.

    Args:
        model_name: ``"cnn"`` ou ``"resnet"``.

    Returns:
        Le modèle Keras chargé.
    """
    import tensorflow as tf

    model_path = MODELS_DIR / f"{model_name}.keras"
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(model_path)


def prepare_image(image_array: np.ndarray) -> np.ndarray:
    """Redimensionne une image et ajoute la dimension de lot.

    La normalisation est gérée à l'intérieur des modèles, l'image reste donc
    en valeurs ``uint8``.

    Args:
        image_array: Image RGB ``uint8``.

    Returns:
        Lot d'une image, prêt pour la prédiction.
    """
    resized = resize_image(image_array, IMAGE_SIZE)
    return np.expand_dims(resized.astype("float32"), axis=0)


def main() -> None:
    """Construit l'interface et gère le cycle de prédiction."""
    st.set_page_config(page_title="Tri intelligent des déchets", page_icon="♻️")
    st.title("♻️ Tri intelligent des déchets")
    st.write(
        "Chargez une image de déchet pour obtenir la classe prédite et le "
        "score de confiance du modèle."
    )

    model_choice = st.sidebar.selectbox(
        "Modèle", options=["resnet", "cnn"], format_func=str.upper
    )
    model = load_trained_model(model_choice)
    if model is None:
        st.error(
            f"Aucun modèle '{model_choice}' trouvé dans models/. "
            "Entraînez-le d'abord avec `python -m src.train`."
        )
        return

    uploaded = st.file_uploader(
        "Image à classifier", type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded is None:
        st.info("En attente d'une image.")
        return

    from PIL import Image

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image chargée", use_container_width=True)

    batch = prepare_image(np.array(image))
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    st.subheader("Résultat")
    st.metric("Classe prédite", CLASS_NAMES[predicted_index].capitalize())
    st.metric("Confiance", f"{confidence:.1%}")

    st.subheader("Probabilités par classe")
    distribution = {
        CLASS_NAMES[i].capitalize(): float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }
    st.bar_chart(distribution)


if __name__ == "__main__":
    main()
