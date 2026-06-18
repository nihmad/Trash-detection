"""Définition du CNN personnalisé.

Architecture volontairement légère afin de rester entraînable sur une
machine personnelle (CPU ou petit GPU). Elle empile plusieurs blocs
convolutifs (Conv2D + BatchNormalization + MaxPooling) suivis d'une tête
dense régularisée par Dropout.
"""

from __future__ import annotations

from src.utils import CLASS_NAMES, IMAGE_SIZE


def build_cnn_model(
    input_shape: tuple[int, int, int] = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    num_classes: int = len(CLASS_NAMES),
    dropout_rate: float = 0.5,
):
    """Construit et compile le CNN personnalisé.

    L'augmentation et la normalisation [0, 1] sont intégrées au modèle, ce
    qui permet de lui fournir directement des images ``uint8``.

    Args:
        input_shape: Forme des images d'entrée ``(H, W, C)``.
        num_classes: Nombre de classes en sortie.
        dropout_rate: Taux de Dropout de la tête dense.

    Returns:
        Un ``tf.keras.Model`` compilé.
    """
    import tensorflow as tf

    from src.preprocessing import build_augmentation_layer

    inputs = tf.keras.Input(shape=input_shape)
    x = build_augmentation_layer()(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    # Trois blocs convolutifs avec un nombre de filtres croissant.
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="custom_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
