"""Tests unitaires du module de prétraitement.

Ces tests créent des images synthétiques sur disque afin de ne dépendre ni
du dataset TrashNet ni de TensorFlow.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    is_image_corrupted,
    load_image,
    normalize_image,
    preprocess_image,
    resize_image,
    scan_corrupted_images,
)


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """Crée une petite image JPEG valide et retourne son chemin."""
    image = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
    path = tmp_path / "sample.jpg"
    cv2.imwrite(str(path), image)
    return path


def test_load_image_returns_rgb(sample_image: Path) -> None:
    """L'image chargée a trois canaux et le bon type."""
    image = load_image(sample_image)
    assert image.shape[2] == 3
    assert image.dtype == np.uint8


def test_resize_image_changes_dimensions(sample_image: Path) -> None:
    """Le redimensionnement produit exactement la taille demandée."""
    image = load_image(sample_image)
    resized = resize_image(image, (32, 32))
    assert resized.shape[:2] == (32, 32)


def test_normalize_image_range(sample_image: Path) -> None:
    """La normalisation borne les pixels dans [0, 1]."""
    image = load_image(sample_image)
    normalized = normalize_image(image)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_preprocess_image_full_pipeline(sample_image: Path) -> None:
    """Le pipeline complet renvoie une image normalisée à la bonne taille."""
    result = preprocess_image(sample_image, (32, 32))
    assert result.shape == (32, 32, 3)
    assert result.max() <= 1.0


def test_is_image_corrupted_on_invalid_file(tmp_path: Path) -> None:
    """Un fichier texte renommé en .jpg est détecté comme corrompu."""
    fake = tmp_path / "broken.jpg"
    fake.write_text("ceci n'est pas une image")
    assert is_image_corrupted(fake) is True


def test_is_image_corrupted_rejects_unknown_extension(tmp_path: Path) -> None:
    """Une extension non supportée est considérée comme invalide."""
    fake = tmp_path / "document.txt"
    fake.write_text("texte")
    assert is_image_corrupted(fake) is True


def test_scan_corrupted_images(tmp_path: Path) -> None:
    """Le scan distingue les images valides des fichiers corrompus."""
    valid = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "ok.png"), valid)
    (tmp_path / "ko.jpg").write_text("corrompu")
    corrupted = scan_corrupted_images(tmp_path)
    assert len(corrupted) == 1
    assert corrupted[0].name == "ko.jpg"
