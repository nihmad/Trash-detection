"""Évaluation des modèles et analyse des erreurs.

Calcule les métriques (accuracy, precision, recall, F1), génère la matrice
de confusion, le rapport de classification et une planche d'exemples mal
classés. Tous les artefacts sont enregistrés dans ``reports/``.

Exemple :

    python -m src.evaluate --model resnet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import build_index, make_tf_dataset, split_dataset
from src.preprocessing import preprocess_image
from src.utils import (
    CLASS_NAMES,
    DATA_RAW_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    ensure_dir,
    get_logger,
    save_json,
    set_global_seed,
)

logger = get_logger(__name__)


def predict_dataframe(model, df: pd.DataFrame) -> np.ndarray:
    """Prédit les probabilités de classe pour un DataFrame d'index.

    Args:
        model: Modèle Keras chargé.
        df: DataFrame avec colonnes ``filepath`` et ``label``.

    Returns:
        Tableau de probabilités de forme ``(n_images, n_classes)``.
    """
    dataset = make_tf_dataset(df, batch_size=32, shuffle=False)
    return model.predict(dataset, verbose=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les métriques globales pondérées.

    Args:
        y_true: Étiquettes réelles (entiers).
        y_pred: Étiquettes prédites (entiers).

    Returns:
        Dictionnaire contenant accuracy, precision, recall et F1.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> Path:
    """Trace et enregistre la matrice de confusion.

    Args:
        y_true: Étiquettes réelles.
        y_pred: Étiquettes prédites.
        model_name: Nom du modèle, pour le nom de fichier.

    Returns:
        Chemin de la figure enregistrée.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    ensure_dir(FIGURES_DIR)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Matrice de confusion - {model_name}")
    fig.tight_layout()

    output_path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Matrice de confusion enregistrée : %s", output_path)
    return output_path


def plot_misclassified(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    max_examples: int = 9,
) -> Path | None:
    """Affiche une planche d'exemples mal classés avec leur score.

    Args:
        df: DataFrame d'index du jeu de test.
        y_true: Étiquettes réelles.
        y_pred: Étiquettes prédites.
        probs: Probabilités prédites.
        model_name: Nom du modèle.
        max_examples: Nombre maximal d'exemples affichés.

    Returns:
        Chemin de la figure, ou ``None`` si aucune erreur.
    """
    import matplotlib.pyplot as plt

    errors = np.where(y_true != y_pred)[0]
    if len(errors) == 0:
        logger.info("Aucune erreur à visualiser pour %s.", model_name)
        return None

    ensure_dir(FIGURES_DIR)
    selected = errors[:max_examples]
    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, selected):
        image = preprocess_image(Path(df.iloc[idx]["filepath"]))
        confidence = float(probs[idx, y_pred[idx]])
        ax.imshow(image)
        ax.set_title(
            f"Réel : {CLASS_NAMES[y_true[idx]]}\n"
            f"Prédit : {CLASS_NAMES[y_pred[idx]]} ({confidence:.0%})",
            fontsize=10,
        )
        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(f"Exemples mal classés - {model_name}")
    fig.tight_layout()
    output_path = FIGURES_DIR / f"{model_name}_misclassified.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Exemples mal classés enregistrés : %s", output_path)
    return output_path


def most_confused_pairs(
    y_true: np.ndarray, y_pred: np.ndarray, top_k: int = 3
) -> list[dict[str, object]]:
    """Identifie les paires de classes les plus confondues.

    Args:
        y_true: Étiquettes réelles.
        y_pred: Étiquettes prédites.
        top_k: Nombre de paires à retourner.

    Returns:
        Liste de dictionnaires ``{vraie, prédite, nombre}`` triée par fréquence.
    """
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    pairs: list[dict[str, object]] = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and matrix[i, j] > 0:
                pairs.append(
                    {
                        "vraie_classe": CLASS_NAMES[i],
                        "classe_predite": CLASS_NAMES[j],
                        "nombre": int(matrix[i, j]),
                    }
                )
    pairs.sort(key=lambda item: item["nombre"], reverse=True)
    return pairs[:top_k]


def evaluate(model_name: str, raw_dir: Path, seed: int) -> dict[str, object]:
    """Évalue un modèle entraîné sur le jeu de test et produit les rapports.

    Args:
        model_name: ``"cnn"`` ou ``"resnet"``.
        raw_dir: Répertoire des images brutes.
        seed: Graine pour reproduire exactement le même découpage.

    Returns:
        Dictionnaire récapitulatif des métriques et de l'analyse d'erreurs.
    """
    import tensorflow as tf
    from sklearn.metrics import classification_report

    set_global_seed(seed)
    model_path = MODELS_DIR / f"{model_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}. Entraînez-le d'abord avec src.train."
        )

    model = tf.keras.models.load_model(model_path)
    index = build_index(raw_dir)
    splits = split_dataset(index, seed=seed)
    test_df = splits.test

    probs = predict_dataframe(model, test_df)
    y_pred = probs.argmax(axis=1)
    class_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_true = test_df["label"].map(class_to_index).to_numpy()

    metrics = compute_metrics(y_true, y_pred)
    logger.info("Métriques %s : %s", model_name, metrics)

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    confused = most_confused_pairs(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_misclassified(test_df, y_true, y_pred, probs, model_name)

    summary = {
        "model": model_name,
        "metrics": metrics,
        "classification_report": report,
        "most_confused_pairs": confused,
        "test_size": int(len(test_df)),
    }
    ensure_dir(METRICS_DIR)
    save_json(summary, METRICS_DIR / f"{model_name}_evaluation.json")
    return summary


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation d'un modèle entraîné.")
    parser.add_argument("--model", choices=["cnn", "resnet"], required=True)
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Point d'entrée du script."""
    args = parse_args()
    evaluate(model_name=args.model, raw_dir=args.raw_dir, seed=args.seed)


if __name__ == "__main__":
    main()
