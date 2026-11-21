"""
Rshishabh - Artificial Neural Network (Keras) + simple tuning.
"""
import keras
import numpy as np
from keras import layers


def build_ann(input_dim, units=(64, 32), dropout=0.2, learning_rate=0.001):
    """Input -> Dense(ReLU, He init) -> Dropout -> Dense(ReLU) -> 1 output."""
    model = keras.Sequential([keras.Input(shape=(input_dim,))])
    for i, n in enumerate(units):
        model.add(layers.Dense(n, activation="relu", kernel_initializer="he_normal"))
        if i == 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))  # regression -> no activation

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])
    return model


def train_ann(X_train, y_train, units=(64, 32), dropout=0.2, learning_rate=0.001, epochs=100):
    """Train with 20% validation split and early stopping."""
    keras.utils.set_random_seed(42)
    model = build_ann(X_train.shape[1], units, dropout, learning_rate)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                               restore_best_weights=True)
    history = model.fit(
        X_train.astype("float32"), y_train.astype("float32"),
        validation_split=0.2, epochs=epochs, batch_size=64,
        callbacks=[early_stop], verbose=0,
    )
    best_val_rmse = float(np.sqrt(min(history.history["val_loss"])))
    print(f"  ANN {units}, dropout={dropout}, lr={learning_rate} "
          f"-> val RMSE {best_val_rmse:.3f} ({len(history.history['loss'])} epochs)")
    return model, history, best_val_rmse


def tune_ann(X_train, y_train):
    """Try a few settings, keep the one with the lowest validation RMSE."""
    configs = [
        {"units": (64, 32), "dropout": 0.2, "learning_rate": 0.001},
        {"units": (128, 64), "dropout": 0.2, "learning_rate": 0.001},
        {"units": (128, 64, 32), "dropout": 0.3, "learning_rate": 0.001},
        {"units": (64, 32), "dropout": 0.1, "learning_rate": 0.01},
    ]
    best = None
    for cfg in configs:
        model, history, val_rmse = train_ann(X_train, y_train, **cfg)
        if best is None or val_rmse < best[2]:
            best = (model, history, val_rmse, cfg)
    print(f"Best ANN config: {best[3]}")
    return best[0], best[1]