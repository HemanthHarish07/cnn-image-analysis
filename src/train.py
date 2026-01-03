import tensorflow as tf


def compile_and_train(
    model,
    train_ds,
    val_ds,
    epochs=10,
    learning_rate=1e-3
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return history
