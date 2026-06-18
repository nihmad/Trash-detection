"""Chargement et préparation du dataset TrashNet.

Responsabilités :
    - Harmoniser les libellés des classes (5 classes retenues).
    - Inventorier les fichiers valides en écartant les images corrompues.
    - Construire des partitions train / validation / test stratifiées.
    - Produire des pipelines ``tf.data`` performants et reproductibles.

Le dataset attendu est organisé en sous-dossiers, un par classe :

    data/raw/
        cardboard/
        glass/
        metal/
        paper/
        plastic/
        trash/   (présent dans TrashNet, ignoré ici)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import is_image_corrupted
from src.utils import CLASS_NAMES, IMAGE_SIZE, get_logger

logger = get_logger(__name__)

# Correspondance entre les noms de dossiers TrashNet et les classes retenues.
# Permet de documenter et de figer l'harmonisation des étiquettes.
LABEL_MAPPING: dict[str, str] = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
}


@dataclass
class DatasetSplits:
    """Conteneur des trois partitions sous forme de DataFrames.

    Chaque DataFrame possède les colonnes ``filepath`` et ``label``.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def build_index(raw_dir: Path, check_corrupted: bool = True) -> pd.DataFrame:
    """Construit un index des images valides à partir des sous-dossiers.

    Args:
        raw_dir: Répertoire contenant un sous-dossier par classe.
        check_corrupted: Si ``True``, écarte les images illisibles.

    Returns:
        DataFrame avec les colonnes ``filepath`` (str) et ``label`` (str).

    Raises:
        FileNotFoundError: Si ``raw_dir`` n'existe pas.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Répertoire introuvable : {raw_dir}. "
            "Téléchargez TrashNet et placez les dossiers de classes dans data/raw/."
        )

    records: list[dict[str, str]] = []
    skipped = 0
    for folder_name, label in LABEL_MAPPING.items():
        class_dir = raw_dir / folder_name
        if not class_dir.is_dir():
            logger.warning("Dossier de classe manquant : %s", class_dir)
            continue
        for path in sorted(class_dir.glob("*")):
            if not path.is_file():
                continue
            if check_corrupted and is_image_corrupted(path):
                skipped += 1
                continue
            records.append({"filepath": str(path), "label": label})

    if skipped:
        logger.warning("%d image(s) ignorée(s) (corrompues/invalides).", skipped)

    index = pd.DataFrame.from_records(records)
    if index.empty:
        raise FileNotFoundError(
            f"Aucune image valide trouvée dans {raw_dir}."
        )
    logger.info(
        "Index construit : %d images, %d classes.",
        len(index),
        index["label"].nunique(),
    )
    return index


def split_dataset(
    index: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DatasetSplits:
    """Découpe l'index en partitions train / validation / test stratifiées.

    La stratification préserve la proportion de chaque classe dans les trois
    partitions.

    Args:
        index: Index produit par :func:`build_index`.
        val_size: Proportion dédiée à la validation.
        test_size: Proportion dédiée au test.
        seed: Graine pour la reproductibilité du découpage.

    Returns:
        Un objet :class:`DatasetSplits`.
    """
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        index,
        test_size=test_size,
        stratify=index["label"],
        random_state=seed,
    )
    # Recalcule la part de validation relative au sous-ensemble restant.
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val["label"],
        random_state=seed,
    )
    logger.info(
        "Découpage : train=%d, val=%d, test=%d.",
        len(train),
        len(val),
        len(test),
    )
    return DatasetSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def _decode_and_resize(filepath, label, image_size: tuple[int, int]):
    """Lit, décode et redimensionne une image dans un graphe TensorFlow.

    Fonction interne utilisée par ``tf.data``. La normalisation est laissée
    aux modèles afin de pouvoir gérer le prétraitement spécifique de ResNet50.
    """
    import tensorflow as tf

    raw = tf.io.read_file(filepath)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image.set_shape((image_size[0], image_size[1], 3))
    return image, label


def make_tf_dataset(
    df: pd.DataFrame,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
):
    """Construit un ``tf.data.Dataset`` à partir d'un DataFrame d'index.

    Args:
        df: DataFrame avec colonnes ``filepath`` et ``label``.
        image_size: Taille cible des images.
        batch_size: Taille des lots.
        shuffle: Active le mélange (utile pour l'entraînement).
        seed: Graine du mélange.

    Returns:
        Un dataset produisant des paires ``(images, labels_entiers)``.
    """
    import tensorflow as tf

    class_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    filepaths = df["filepath"].to_numpy()
    labels = df["label"].map(class_to_index).to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=seed)
    dataset = dataset.map(
        lambda fp, lb: _decode_and_resize(fp, lb, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_distribution(index: pd.DataFrame) -> pd.Series:
    """Retourne le nombre d'images par classe.

    Args:
        index: Index des images.

    Returns:
        Série indexée par nom de classe, triée par ordre alphabétique.
    """
    return index["label"].value_counts().sort_index()
