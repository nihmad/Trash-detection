"""Pipeline de prétraitement d'images basé sur OpenCV.

Étapes couvertes :
    1. Vérification des images corrompues.
    2. Lecture et conversion de format (BGR -> RGB).
    3. Redimensionnement à une taille cible.
    4. Normalisation des pixels dans l'intervalle [0, 1].
    5. Configuration de l'augmentation de données (rotation, zoom,
       translation, flip horizontal).

Chaque fonction est indépendante afin de pouvoir être testée et réutilisée.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import IMAGE_SIZE, get_logger

logger = get_logger(__name__)

# Extensions d'images considérées comme valides.
VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_corrupted(path: Path) -> bool:
    """Détermine si une image est illisible ou corrompue.

    L'image est ouverte avec OpenCV ; un retour ``None`` ou une exception
    indique un fichier inutilisable.

    Args:
        path: Chemin du fichier image.

    Returns:
        ``True`` si l'image est corrompue ou illisible, ``False`` sinon.
    """
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return True
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except cv2.error:
        return True
    return image is None or image.size == 0


def load_image(path: Path) -> np.ndarray:
    """Charge une image et la convertit de BGR (OpenCV) vers RGB.

    Args:
        path: Chemin du fichier image.

    Returns:
        Image RGB sous forme de tableau ``uint8`` de forme (H, W, 3).

    Raises:
        ValueError: Si l'image ne peut pas être lue.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible : {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    """Redimensionne une image à la taille cible.

    L'interpolation par aire est utilisée pour la réduction car elle limite
    le crénelage par rapport à une interpolation linéaire.

    Args:
        image: Image source.
        size: Taille cible ``(largeur, hauteur)``.

    Returns:
        Image redimensionnée.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Met les pixels à l'échelle dans l'intervalle [0, 1].

    Args:
        image: Image ``uint8`` (valeurs 0-255).

    Returns:
        Image ``float32`` normalisée.
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(
    path: Path, size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    """Applique la chaîne complète de prétraitement à une image.

    Enchaîne lecture, conversion RGB, redimensionnement et normalisation.

    Args:
        path: Chemin du fichier image.
        size: Taille cible ``(largeur, hauteur)``.

    Returns:
        Image normalisée prête à être fournie à un modèle.
    """
    image = load_image(path)
    image = resize_image(image, size)
    return normalize_image(image)


def scan_corrupted_images(directory: Path) -> list[Path]:
    """Parcourt récursivement un répertoire et liste les images corrompues.

    Args:
        directory: Répertoire racine à inspecter.

    Returns:
        Liste des chemins d'images détectées comme corrompues.
    """
    corrupted: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and is_image_corrupted(path):
            corrupted.append(path)
    if corrupted:
        logger.warning("%d image(s) corrompue(s) détectée(s).", len(corrupted))
    else:
        logger.info("Aucune image corrompue détectée dans %s.", directory)
    return corrupted


def build_augmentation_layer():
    """Construit une couche d'augmentation de données Keras.

    Couvre les transformations demandées : rotation, zoom, translation et
    retournement horizontal. L'augmentation est appliquée uniquement à
    l'entraînement (les couches sont inactives en inférence).

    Returns:
        Un ``tf.keras.Sequential`` regroupant les couches d'augmentation.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ],
        name="data_augmentation",
    )
