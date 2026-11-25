import os
import sys

# Suppress TensorFlow C++ logging warnings (INFO, WARNING, ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
# Option 1: Load standard MNIST dataset via Keras
from tensorflow.keras.datasets import mnist

# =======================================================
# 1. Vectorized Backpropagation Neural Network
# =======================================================
class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=64, output_size=10, learning_rate=0.1):
        self.input_size = input_size       # 28x28 pixels = 784 inputs
        self.hidden_size = hidden_size     # 64 hidden neurons
        self.output_size = output_size     # 10 classes (digits 0-9)
        self.lr = learning_rate

        # Seed random weights for reproducibility
        np.random.seed(42)
        self.W1 = np.random.uniform(-0.1, 0.1, (self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        
        self.W2 = np.random.uniform(-0.1, 0.1, (self.hidden_size, self.output_size))
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1.0 - a)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def train_batch(self, X_batch, y_batch):
        m = X_batch.shape[0]

        # 1. Forward Pass
        output = self.forward(X_batch)

        # 2. Output Layer Deltas
        error_output = y_batch - output
        delta_output = error_output * self.sigmoid_derivative(output)

        # 3. Hidden Layer Deltas
        error_hidden = np.dot(delta_output, self.W2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # 4. Weight & Bias Updates
        self.W2 += (self.lr / m) * np.dot(self.a1.T, delta_output)
        self.b2 += (self.lr / m) * np.sum(delta_output, axis=0, keepdims=True)

        self.W1 += (self.lr / m) * np.dot(X_batch.T, delta_hidden)
        self.b1 += (self.lr / m) * np.sum(delta_hidden, axis=0, keepdims=True)

        return np.mean(np.square(error_output))

    def predict(self, X):
        output = self.forward(X)
        predicted_class = np.argmax(output, axis=1)[0]
        confidence = output[0, predicted_class] * 100.0
        return predicted_class, confidence, output[0]

    def save_weights(self, filepath="model_weights.npz"):
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"[+] Model weights saved to '{filepath}'")

    def load_weights(self, filepath="model_weights.npz"):
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            print(f"[+] Model weights loaded successfully from '{filepath}'")
            return True
        return False


# =======================================================
# 2. Preprocessing for Paper Photos (28x28 Output)
# =======================================================
def preprocess_paper_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return None

    # 1. Load image in grayscale & blur
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. Binary Inverted Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Dilate stroke to solidify lines while keeping internal loops open
    kernel = np.ones((3, 3), np.uint8)
    thresh_dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 4. Crop bounding box around digit
    contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: No stroke detected.")
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Aspect ratio preserving resize into a 20x20 box (standard MNIST inner region)
    max_dim = max(w, h)
    scale = 20.0 / max_dim
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    
    digit_crop = thresh_dilated[y:y+h, x:x+w]
    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place resized digit into a 28x28 canvas using Center of Mass
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate Moments to find Center of Mass of the stroke
    M = cv2.moments(resized_digit)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = new_w // 2, new_h // 2

    # Shift canvas so the center of mass rests at (14, 14)
    start_x = max(0, min(28 - new_w, 14 - cx))
    start_y = max(0, min(28 - new_h, 14 - cy))

    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit

    # Save visual debug output
    cv2.imwrite("debug_input.png", canvas)

    return (canvas / 255.0).reshape(1, 784)


# =======================================================
# 3. Main Training & Inference Pipeline
# =======================================================
if __name__ == "__main__":
    print("=======================================================")
    print(" 28x28 Backpropagation Network with Real MNIST Dataset")
    print("=======================================================")

    weights_file = "model_weights.npz"
    nn = NeuralNetwork(input_size=784, hidden_size=64, output_size=10, learning_rate=0.5)

    # 1. Load weights if available; train ONLY if file doesn't exist
    if nn.load_weights(weights_file):
        print("[+] Skipping training loop, model is ready for instant inference!")
    else:
        # Fetch MNIST Dataset via Keras
        print("\n[+] Downloading / Loading MNIST dataset...")
        (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

        # Reshape & Normalize inputs to range [0.0, 1.0]
        X_train = X_train_raw.reshape(-1, 784) / 255.0
        X_test = X_test_raw.reshape(-1, 784) / 255.0

        # One-hot encode targets (Full 60,000 training set)
        num_train = 60000
        y_train = np.zeros((num_train, 10))
        for i in range(num_train):
            y_train[i, y_train_raw[i]] = 1.0

        # Train in Mini-Batches
        batch_size = 64
        epochs = 30
        print(f"\n[+] Training network for {epochs} epochs (Batch size: {batch_size})...")

        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(num_train)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            epoch_loss = 0.0
            num_batches = num_train // batch_size

            for b in range(num_batches):
                start = b * batch_size
                end = start + batch_size
                loss = nn.train_batch(X_shuffled[start:end], y_shuffled[start:end])
                epoch_loss += loss

            print(f"Epoch {epoch:2d}/{epochs} | Mean Loss: {epoch_loss / num_batches:.6f}")

        # Test set evaluation
        test_outputs = nn.forward(X_test)
        predictions = np.argmax(test_outputs, axis=1)
        accuracy = np.mean(predictions == y_test_raw) * 100.0
        print(f"\n[+] Model Accuracy on MNIST Test Set: {accuracy:.1f}%")

        # Save weights so subsequent runs load immediately
        nn.save_weights(weights_file)

    # 2. Evaluate paper photo if provided in terminal
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        print(f"\n[+] Testing image file: {image_file}")
        
        input_vector = preprocess_paper_image(image_file)
        if input_vector is not None:
            digit, conf, scores = nn.predict(input_vector)
            
            print("\nOutput Layer Activations:")
            for k in range(10):
                print(f" Class [{k}]: {scores[k]:.4f}")

            print(f"\nFinal Result -> Digit Recognized: {digit} (Confidence: {conf:.1f}%)")
