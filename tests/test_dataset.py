"""Tests unitaires du module de gestion du dataset.

Un mini-dataset synthétique est généré sur disque pour valider l'indexation,
le découpage stratifié et la distribution des classes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dataset import (
    build_index,
    class_distribution,
    split_dataset,
)
from src.utils import CLASS_NAMES


@pytest.fixture()
def fake_dataset(tmp_path: Path) -> Path:
    """Crée un dataset jouet avec quelques images par classe."""
    for label in CLASS_NAMES:
        class_dir = tmp_path / label
        class_dir.mkdir()
        for i in range(10):
            image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"{label}_{i}.jpg"), image)
    return tmp_path


def test_build_index_counts_all_images(fake_dataset: Path) -> None:
    """L'index recense toutes les images valides du dataset jouet."""
    index = build_index(fake_dataset)
    assert len(index) == len(CLASS_NAMES) * 10
    assert set(index["label"].unique()) == set(CLASS_NAMES)


def test_build_index_missing_dir() -> None:
    """Un répertoire inexistant déclenche une erreur explicite."""
    with pytest.raises(FileNotFoundError):
        build_index(Path("/chemin/inexistant"))


def test_split_dataset_proportions(fake_dataset: Path) -> None:
    """Les trois partitions sont disjointes et couvrent tout l'index."""
    index = build_index(fake_dataset)
    splits = split_dataset(index, val_size=0.2, test_size=0.2, seed=0)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == len(index)
    train_paths = set(splits.train["filepath"])
    test_paths = set(splits.test["filepath"])
    assert train_paths.isdisjoint(test_paths)


def test_class_distribution(fake_dataset: Path) -> None:
    """La distribution renvoie le nombre attendu d'images par classe."""
    index = build_index(fake_dataset)
    distribution = class_distribution(index)
    assert all(count == 10 for count in distribution)
