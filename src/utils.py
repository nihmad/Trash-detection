"""Fonctions utilitaires partagées par l'ensemble du projet.

Regroupe la configuration du logging, la fixation des graines aléatoires,
la gestion centralisée des chemins et la lecture/écriture de fichiers JSON.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

# Racine du projet, déduite à partir de l'emplacement de ce fichier.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Chemins de référence utilisés par les différents scripts.
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

# Classes harmonisées du projet (la classe "trash" de TrashNet est exclue).
CLASS_NAMES: list[str] = ["cardboard", "glass", "metal", "paper", "plastic"]

# Taille d'entrée commune aux deux modèles.
IMAGE_SIZE: tuple[int, int] = (128, 128)


def set_global_seed(seed: int = 42) -> None:
    """Fixe les graines aléatoires pour rendre les exécutions reproductibles.

    Args:
        seed: Valeur utilisée pour Python, NumPy et TensorFlow.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        # TensorFlow n'est pas toujours nécessaire (ex. tests de prétraitement).
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Retourne un logger configuré avec un format lisible.

    Args:
        name: Nom du logger, généralement ``__name__``.
        level: Niveau de journalisation minimal.

    Returns:
        Une instance de :class:`logging.Logger` prête à l'emploi.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def ensure_dir(path: Path) -> Path:
    """Crée un répertoire s'il n'existe pas et retourne son chemin.

    Args:
        path: Répertoire à créer.

    Returns:
        Le chemin du répertoire, désormais garanti existant.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: Path) -> None:
    """Sérialise un dictionnaire au format JSON indenté.

    Args:
        data: Données à enregistrer.
        path: Fichier de destination.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    """Charge un fichier JSON.

    Args:
        path: Fichier à lire.

    Returns:
        Le contenu désérialisé.
    """
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
