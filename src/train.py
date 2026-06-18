"""Script d'entraînement des deux modèles.

Exemples d'utilisation :

    python -m src.train --model cnn --epochs 30
    python -m src.train --model resnet --epochs 15 --fine-tune-epochs 10

Le script construit l'index du dataset, crée les partitions, entraîne le
modèle choisi, sauvegarde les poids et l'historique d'entraînement, puis
génère les courbes d'apprentissage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import build_index, make_tf_dataset, split_dataset
from src.utils import (
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


def plot_history(history: dict[str, list[float]], model_name: str) -> Path:
    """Trace et sauvegarde les courbes d'entraînement et de validation.

    Args:
        history: Dictionnaire des métriques par époque.
        model_name: Nom du modèle, utilisé pour nommer le fichier.

    Returns:
        Chemin de la figure enregistrée.
    """
    import matplotlib.pyplot as plt

    ensure_dir(FIGURES_DIR)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="validation")
    axes[0].set_title(f"Accuracy - {model_name}")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="validation")
    axes[1].set_title(f"Loss - {model_name}")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    output_path = FIGURES_DIR / f"{model_name}_training_curves.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Courbes d'entraînement enregistrées : %s", output_path)
    return output_path


def merge_histories(
    first: dict[str, list[float]], second: dict[str, list[float]]
) -> dict[str, list[float]]:
    """Concatène deux historiques Keras (phase tête + phase fine-tuning).

    Args:
        first: Historique de la première phase.
        second: Historique de la seconde phase.

    Returns:
        Historique fusionné, clé par clé.
    """
    merged: dict[str, list[float]] = {}
    for key in first:
        merged[key] = list(first[key]) + list(second.get(key, []))
    return merged


def build_callbacks(model_path: Path):
    """Crée les callbacks d'entraînement (early stopping, sauvegarde, LR).

    Args:
        model_path: Chemin de sauvegarde du meilleur modèle.

    Returns:
        Liste de callbacks Keras.
    """
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]


def train(
    model_name: str,
    epochs: int,
    fine_tune_epochs: int,
    batch_size: int,
    raw_dir: Path,
    seed: int,
) -> None:
    """Entraîne le modèle demandé et enregistre les artefacts.

    Args:
        model_name: ``"cnn"`` ou ``"resnet"``.
        epochs: Nombre d'époques de la phase principale.
        fine_tune_epochs: Époques de fine-tuning (ResNet50 uniquement).
        batch_size: Taille des lots.
        raw_dir: Répertoire du dataset brut.
        seed: Graine de reproductibilité.
    """
    set_global_seed(seed)
    ensure_dir(MODELS_DIR)

    index = build_index(raw_dir)
    splits = split_dataset(index, seed=seed)
    train_ds = make_tf_dataset(
        splits.train, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_ds = make_tf_dataset(splits.val, batch_size=batch_size)

    model_path = MODELS_DIR / f"{model_name}.keras"

    if model_name == "cnn":
        from src.cnn_model import build_cnn_model

        model = build_cnn_model()
        model.summary(print_fn=logger.info)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=build_callbacks(model_path),
        )
        full_history = history.history

    elif model_name == "resnet":
        from src.resnet_model import build_resnet_model, unfreeze_for_fine_tuning

        model = build_resnet_model()
        model.summary(print_fn=logger.info)

        logger.info("Phase 1 : entraînement de la tête (base gelée).")
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=build_callbacks(model_path),
        )

        logger.info("Phase 2 : fine-tuning des dernières couches.")
        model = unfreeze_for_fine_tuning(model)
        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=build_callbacks(model_path),
        )
        full_history = merge_histories(head_history.history, ft_history.history)

    else:
        raise ValueError(f"Modèle inconnu : {model_name!r} (cnn ou resnet).")

    model.save(model_path)
    logger.info("Modèle sauvegardé : %s", model_path)

    ensure_dir(METRICS_DIR)
    history_path = METRICS_DIR / f"{model_name}_history.json"
    save_json(full_history, history_path)
    plot_history(full_history, model_name)


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement des modèles de tri.")
    parser.add_argument(
        "--model", choices=["cnn", "resnet"], required=True, help="Modèle à entraîner."
    )
    parser.add_argument("--epochs", type=int, default=30, help="Époques principales.")
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=10,
        help="Époques de fine-tuning (ResNet50).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Taille des lots.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help="Répertoire des images brutes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée du script."""
    args = parse_args()
    train(
        model_name=args.model,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        raw_dir=args.raw_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
