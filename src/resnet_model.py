"""Modèle de transfert d'apprentissage basé sur ResNet50.

Stratégie en deux temps :
    1. Entraînement initial avec la base convolutive gelée (seule la tête
       de classification apprend).
    2. Fine-tuning : dégel des dernières couches de la base pour un
       réglage fin à faible taux d'apprentissage.

Le prétraitement spécifique à ResNet50 (``preprocess_input``) est intégré
au modèle pour rester cohérent avec les poids ImageNet.
"""

from __future__ import annotations

from src.utils import CLASS_NAMES, IMAGE_SIZE


def build_resnet_model(
    input_shape: tuple[int, int, int] = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    num_classes: int = len(CLASS_NAMES),
    dropout_rate: float = 0.4,
):
    """Construit et compile le modèle ResNet50 avec base gelée.

    Args:
        input_shape: Forme des images d'entrée ``(H, W, C)``.
        num_classes: Nombre de classes en sortie.
        dropout_rate: Taux de Dropout de la tête de classification.

    Returns:
        Un ``tf.keras.Model`` compilé, base convolutive gelée.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input

    from src.preprocessing import build_augmentation_layer

    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = build_augmentation_layer()(inputs)
    # ResNet50 attend des entrées centrées selon les statistiques ImageNet.
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="resnet50_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_fine_tuning(
    model,
    num_layers: int = 30,
    learning_rate: float = 1e-5,
):
    """Dégèle les dernières couches de la base et recompile pour le fine-tuning.

    Le modèle doit avoir été créé par :func:`build_resnet_model`. Les couches
    de BatchNormalization sont laissées gelées pour stabiliser l'entraînement.

    Args:
        model: Modèle ResNet50 préalablement entraîné tête seule.
        num_layers: Nombre de couches finales de la base à rendre entraînables.
        learning_rate: Taux d'apprentissage réduit pour le réglage fin.

    Returns:
        Le modèle recompilé, prêt pour le fine-tuning.
    """
    import tensorflow as tf

    # Récupère la sous-couche ResNet50 imbriquée dans le modèle fonctionnel.
    base_model = next(
        layer for layer in model.layers if layer.name == "resnet50"
    )
    base_model.trainable = True

    # Ne dégèle que les dernières couches, et garde la BatchNorm gelée.
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
