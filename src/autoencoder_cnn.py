import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_flat = x_train.reshape(-1, 784) / 255.0
x_test_flat = x_test.reshape(-1, 784) / 255.0

print("\n--- Accuracy vs Autoencoder Compression ---")

# Function to build autoencoder model
def build_autoencoder(input_dim, bottleneck_dim):
    input_img = Input(shape=(input_dim,))
    encoded = Dense(bottleneck_dim, activation='relu')(input_img)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder, encoder

for bottleneck_size in [16, 32, 64, 128, 256]:
    # Build & train
    autoencoder, encoder = build_autoencoder(784, bottleneck_size)
    autoencoder.fit(
        x_train_flat, x_train_flat,
        epochs=10,
        batch_size=256,
        shuffle=True,
        verbose=0,
        validation_data=(x_test_flat, x_test_flat)
    )

    # Encode & decode
    x_train_encoded = encoder.predict(x_train_flat)
    x_test_encoded = encoder.predict(x_test_flat)

    x_test_decoded = autoencoder.predict(x_test_flat)

    # Classification
    clf = LogisticRegression(max_iter=3000)
    clf.fit(x_train_encoded, y_train)
    acc = clf.score(x_test_encoded, y_test)
    print(f"Autoencoder({bottleneck_size}) → Accuracy: {acc:.4f}")

    if bottleneck_size == 64:
        # Visualize only for one of them
        def show_reconstructions(n=10):
            indices = random.sample(range(len(x_test)), n)
            plt.figure(figsize=(10, 4))
            for i, idx in enumerate(indices):
                # Original
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(x_test[idx], cmap='gray')
                ax.axis('off')

                # Reconstructed
                ax = plt.subplot(2, n, i + 1 + n)
                recon = x_test_decoded[idx].reshape(28, 28)
                plt.imshow(recon, cmap='gray')
                ax.axis('off')

            plt.suptitle(f"Autoencoder {bottleneck_size} → Original vs Reconstructed", fontsize=14)
            plt.tight_layout()
            plt.show()

        show_reconstructions()

# Final full comparison
clf_original = LogisticRegression(max_iter=3000)
clf_original.fit(x_train_flat, y_train)
y_pred_original = clf_original.predict(x_test_flat)
acc_original = accuracy_score(y_test, y_pred_original)

print(f"\nAccuracy with original features: {acc_original:.4f}")
