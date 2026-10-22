import re
import string
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import Literal

import click
import keras
import tensorflow as tf
from keras import layers

DATASET_SEED_SPLIT = 1234
BATCH_SIZE = 32

# Model constants
MAX_FEATURES = 20_000
EMBEDDING_DIM = 128
SEQUENCE_LENGTH = 500

VALIDATION_SPLIT = 0.2
LABEL_MODE = "binary"

EPOCHS = 3

# Export model constants
LABELS = ("neg", "pos")
MODEL_PATH = "imdb_review.keras"


@click.group()
def main() -> None:
    pass


def setup_tf[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices(gpus[0], "GPU")
        return func(*args, **kwargs)

    return wrapped


def load_ds(
    path: str | Path,
    subset: Literal["training", "validation", "test"],
    class_names: Sequence[str] | None = None,
) -> tf.data.Dataset:
    if subset == "test":
        return keras.utils.text_dataset_from_directory(
            path,
            batch_size=BATCH_SIZE,
            label_mode=LABEL_MODE,
            class_names=class_names,
        )

    return keras.utils.text_dataset_from_directory(
        path,
        batch_size=BATCH_SIZE,
        label_mode=LABEL_MODE,
        validation_split=VALIDATION_SPLIT,
        subset=subset,
        class_names=class_names,
        seed=DATASET_SEED_SPLIT,
    )


@keras.saving.register_keras_serializable(
    package="Custom",
    name="custom_standardization",
)
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html,
        f"[{re.escape(string.punctuation)}]",
        "",
    )


def build_vectorize_layer(adapt_ds: tf.data.Dataset) -> layers.TextVectorization:
    vectorize_layer = layers.TextVectorization(
        max_tokens=MAX_FEATURES,
        standardize=custom_standardization,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    vectorize_layer.adapt(adapt_ds)

    return vectorize_layer


def build_and_train_model(
    vectorize_layer: layers.TextVectorization,
    raw_train_ds: tf.data.Dataset,
    raw_val_ds: tf.data.Dataset,
    raw_test_ds: tf.data.Dataset,
) -> keras.Model:
    # Vectorize text function
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    # Vectorize datasets
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # Prefetch data
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    test_ds = test_ds.cache().prefetch(buffer_size=10)

    # Build model
    inputs = keras.Input(shape=(None,), dtype="int64")

    # Embedding layers
    x = layers.Embedding(MAX_FEATURES, EMBEDDING_DIM)(inputs)
    x = layers.Dropout(0.5)(x)
    # x = layers.GlobalAveragePooling1D()(x)

    # Convolutional layers
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPool1D()(x)

    # Vanilla hidden layer
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # Prediction layer
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = keras.Model(inputs, predictions)

    # Compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    # Train model
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.evaluate(test_ds)

    return model


def build_e2e_model(
    model: keras.Model,
    vectorize_layer: layers.TextVectorization,
) -> keras.Model:
    inputs = keras.Input(shape=(1,), dtype=tf.string, name="text")
    indices = vectorize_layer(inputs)
    outputs = model(indices)

    e2e_model = keras.Model(inputs, outputs)
    e2e_model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return e2e_model


@main.command("build")
@click.option("--ds-path", default="aclImdb")
@click.option("--output", "model_save_path", default=MODEL_PATH)
@setup_tf
def build_and_train_model_e2e_pipeline(ds_path: str, model_save_path: str) -> None:
    # Show banner
    msg = "Keras & TensorFlow learning project."
    msg_len = len(msg)

    def div() -> None:
        click.echo("=" * msg_len)

    div()
    click.echo(msg)
    div()

    # Load datasets
    raw_train_ds = load_ds(
        path=Path(ds_path) / "train",
        subset="training",
        class_names=LABELS,
    )
    raw_val_ds = load_ds(
        path=Path(ds_path) / "train",
        subset="validation",
        class_names=LABELS,
    )
    raw_test_ds = load_ds(
        path=Path(ds_path) / "test",
        subset="test",
        class_names=LABELS,
    )

    click.echo()
    click.echo(f"Training:   {raw_train_ds.cardinality().numpy()}")
    click.echo(f"Validation: {raw_val_ds.cardinality().numpy()}")
    click.echo(f"Test:       {raw_test_ds.cardinality().numpy()}")
    click.echo()

    # Vectorize layer
    vectorize_layer = build_vectorize_layer(
        adapt_ds=raw_train_ds.map(lambda x, _y: x),
    )

    model = build_and_train_model(
        vectorize_layer=vectorize_layer,
        raw_train_ds=raw_train_ds,
        raw_val_ds=raw_val_ds,
        raw_test_ds=raw_test_ds,
    )

    e2e_model = build_e2e_model(model=model, vectorize_layer=vectorize_layer)
    e2e_model.evaluate(raw_test_ds)

    e2e_model.save(filepath=model_save_path)

    div()
    click.echo(f"The trained model has saved at {model_save_path}")


@main.command("run")
@click.option(
    "--model",
    "model_path",
    default=MODEL_PATH,
)
@click.argument("review-text")
@setup_tf
def run(model_path: str, review_text: str) -> None:
    click.echo(f"Text: {review_text}")

    model = keras.saving.load_model(
        model_path,
        custom_objects={
            "custom_standardization": custom_standardization,
        },
        compile=False,
    )

    result = model.predict(tf.constant([[review_text]]), verbose=0)
    score = float(result[0, 0])
    click.echo(f"Result: {score:.3f} ({'positive' if score >= 0.5 else 'negative'})")


if __name__ == "__main__":
    main()
