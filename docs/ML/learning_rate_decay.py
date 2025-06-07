 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a//dev/null b/docs/ML/learning_rate_decay.py
index 0000000000000000000000000000000000000000..0acb5090f11d9ecaf2ce687886be6112370babdf 100644
--- a//dev/null
+++ b/docs/ML/learning_rate_decay.py
@@ -0,0 +1,274 @@
+"""Demonstration of scheduled learning rate decay on a simple
+three-layer neural network using NumPy.
+
+The model can be trained with standard gradient descent, momentum, or
+Adam. A decay function can be passed to gradually decrease the learning
+rate during training.
+
+This script is intended to illustrate the effect of learning rate decay
+with different optimizers. It is not optimized for performance.
+"""
+
+import numpy as np
+import matplotlib.pyplot as plt
+from sklearn.datasets import make_moons
+from sklearn.model_selection import train_test_split
+
+
+def initialize_parameters(layer_dims, seed=42):
+    """Initialize parameters for a fully connected neural network."""
+    np.random.seed(seed)
+    parameters = {}
+    L = len(layer_dims) - 1
+    for l in range(1, L + 1):
+        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
+        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
+    return parameters
+
+
+def linear_forward(A, W, b):
+    Z = W @ A + b
+    cache = (A, W, b)
+    return Z, cache
+
+
+def relu(Z):
+    return np.maximum(0, Z)
+
+
+def relu_backward(dA, Z):
+    dZ = np.array(dA, copy=True)
+    dZ[Z <= 0] = 0
+    return dZ
+
+
+def sigmoid(Z):
+    return 1 / (1 + np.exp(-Z))
+
+
+def sigmoid_backward(dA, Z):
+    s = sigmoid(Z)
+    return dA * s * (1 - s)
+
+
+def forward_propagation(X, parameters):
+    caches = []
+    A = X
+    L = len(parameters) // 2
+    for l in range(1, L):
+        A_prev = A
+        Z, linear_cache = linear_forward(A_prev, parameters[f"W{l}"], parameters[f"b{l}"])
+        A = relu(Z)
+        caches.append((linear_cache, Z))
+    ZL, linear_cache = linear_forward(A, parameters[f"W{L}"], parameters[f"b{L}"])
+    AL = sigmoid(ZL)
+    caches.append((linear_cache, ZL))
+    return AL, caches
+
+
+def compute_cost(AL, Y):
+    m = Y.shape[1]
+    cost = - (1 / m) * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
+    return cost
+
+
+def backward_propagation(AL, Y, caches):
+    grads = {}
+    L = len(caches)
+    m = AL.shape[1]
+    Y = Y.reshape(AL.shape)
+    dAL = - (np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))
+
+    current_cache = caches[L - 1]
+    linear_cache, ZL = current_cache
+    dZL = sigmoid_backward(dAL, ZL)
+    A_prev, W, _ = linear_cache
+    grads[f"dW{L}"] = (1 / m) * dZL @ A_prev.T
+    grads[f"db{L}"] = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
+    dA_prev = W.T @ dZL
+
+    for l in reversed(range(L - 1)):
+        current_cache = caches[l]
+        linear_cache, Z = current_cache
+        dZ = relu_backward(dA_prev, Z)
+        A_prev, W, _ = linear_cache
+        grads[f"dW{l + 1}"] = (1 / m) * dZ @ A_prev.T
+        grads[f"db{l + 1}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
+        dA_prev = W.T @ dZ
+    return grads
+
+
+def initialize_velocity(parameters):
+    L = len(parameters) // 2
+    v = {}
+    for l in range(1, L + 1):
+        v[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
+        v[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
+    return v
+
+
+def update_parameters_with_gd(parameters, grads, learning_rate):
+    L = len(parameters) // 2
+    for l in range(1, L + 1):
+        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
+        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
+    return parameters
+
+
+def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
+    L = len(parameters) // 2
+    for l in range(1, L + 1):
+        v[f"dW{l}"] = beta * v[f"dW{l}"] + (1 - beta) * grads[f"dW{l}"]
+        v[f"db{l}"] = beta * v[f"db{l}"] + (1 - beta) * grads[f"db{l}"]
+        parameters[f"W{l}"] -= learning_rate * v[f"dW{l}"]
+        parameters[f"b{l}"] -= learning_rate * v[f"db{l}"]
+    return parameters, v
+
+
+def initialize_adam(parameters):
+    L = len(parameters) // 2
+    v = {}
+    s = {}
+    for l in range(1, L + 1):
+        v[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
+        v[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
+        s[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
+        s[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
+    return v, s
+
+
+def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate,
+                                beta1, beta2, epsilon=1e-8):
+    L = len(parameters) // 2
+    v_corrected = {}
+    s_corrected = {}
+
+    for l in range(1, L + 1):
+        v[f"dW{l}"] = beta1 * v[f"dW{l}"] + (1 - beta1) * grads[f"dW{l}"]
+        v[f"db{l}"] = beta1 * v[f"db{l}"] + (1 - beta1) * grads[f"db{l}"]
+
+        v_corrected[f"dW{l}"] = v[f"dW{l}"] / (1 - beta1 ** t)
+        v_corrected[f"db{l}"] = v[f"db{l}"] / (1 - beta1 ** t)
+
+        s[f"dW{l}"] = beta2 * s[f"dW{l}"] + (1 - beta2) * (grads[f"dW{l}"] ** 2)
+        s[f"db{l}"] = beta2 * s[f"db{l}"] + (1 - beta2) * (grads[f"db{l}"] ** 2)
+
+        s_corrected[f"dW{l}"] = s[f"dW{l}"] / (1 - beta2 ** t)
+        s_corrected[f"db{l}"] = s[f"db{l}"] / (1 - beta2 ** t)
+
+        parameters[f"W{l}"] -= learning_rate * (
+            v_corrected[f"dW{l}"] / (np.sqrt(s_corrected[f"dW{l}"] + epsilon))
+        )
+        parameters[f"b{l}"] -= learning_rate * (
+            v_corrected[f"db{l}"] / (np.sqrt(s_corrected[f"db{l}"] + epsilon))
+        )
+    return parameters, v, s
+
+
+def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
+    m = X.shape[1]
+    mini_batches = []
+    np.random.seed(seed)
+    permutation = np.random.permutation(m)
+    shuffled_X = X[:, permutation]
+    shuffled_Y = Y[:, permutation]
+    num_complete_minibatches = m // mini_batch_size
+    for k in range(num_complete_minibatches):
+        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
+        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
+        mini_batches.append((mini_batch_X, mini_batch_Y))
+    if m % mini_batch_size != 0:
+        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
+        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
+        mini_batches.append((mini_batch_X, mini_batch_Y))
+    return mini_batches
+
+
+def time_based_decay(initial_lr, epoch, decay_rate):
+    return initial_lr / (1 + decay_rate * epoch)
+
+
+def step_decay(initial_lr, epoch, decay_rate, drop_every=1000):
+    return initial_lr * (decay_rate ** (epoch // drop_every))
+
+
+def exponential_decay(initial_lr, epoch, decay_rate):
+    return initial_lr * np.exp(-decay_rate * epoch)
+
+
+def model(X, Y, layers_dims, optimizer, learning_rate=0.01, mini_batch_size=64,
+          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000,
+          print_cost=False, decay=None, decay_rate=1):
+    L = len(layers_dims)
+    costs = []
+    t = 0
+    seed = 10
+    m = X.shape[1]
+    learning_rate0 = learning_rate
+
+    parameters = initialize_parameters(layers_dims)
+
+    if optimizer == "gd":
+        pass
+    elif optimizer == "momentum":
+        v = initialize_velocity(parameters)
+    elif optimizer == "adam":
+        v, s = initialize_adam(parameters)
+
+    for i in range(num_epochs):
+        seed += 1
+        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
+        cost_total = 0
+
+        for minibatch in minibatches:
+            minibatch_X, minibatch_Y = minibatch
+            AL, caches = forward_propagation(minibatch_X, parameters)
+            cost_total += compute_cost(AL, minibatch_Y)
+            grads = backward_propagation(AL, minibatch_Y, caches)
+
+            if optimizer == "gd":
+                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
+            elif optimizer == "momentum":
+                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
+            elif optimizer == "adam":
+                t += 1
+                parameters, v, s = update_parameters_with_adam(
+                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon
+                )
+        cost_avg = cost_total / len(minibatches)
+        if decay:
+            learning_rate = decay(learning_rate0, i, decay_rate)
+        if print_cost and i % 1000 == 0:
+            print(f"Cost after epoch {i}: {cost_avg:.4f}")
+            if decay:
+                print(f"Learning rate after epoch {i}: {learning_rate:.6f}")
+        if print_cost and i % 100 == 0:
+            costs.append(cost_avg)
+
+    plt.plot(costs)
+    plt.ylabel("cost")
+    plt.xlabel("epochs (per 100)")
+    plt.title(f"Learning rate = {learning_rate}")
+    plt.show()
+    return parameters
+
+
+if __name__ == "__main__":
+    X, Y = make_moons(n_samples=1000, noise=0.2, random_state=1)
+    X = X.T
+    Y = Y.reshape(1, -1)
+
+    layers = [X.shape[0], 5, 2, 1]
+
+    print("Training with gradient descent and time-based decay...")
+    model(X, Y, layers, optimizer="gd", learning_rate=0.05,
+          num_epochs=3000, decay=time_based_decay, decay_rate=0.01)
+
+    print("Training with momentum and step decay...")
+    model(X, Y, layers, optimizer="momentum", learning_rate=0.05,
+          num_epochs=3000, decay=lambda lr, e, dr: step_decay(lr, e, dr, 500),
+          decay_rate=0.5)
+
+    print("Training with Adam and exponential decay...")
+    model(X, Y, layers, optimizer="adam", learning_rate=0.05,
+          num_epochs=3000, decay=exponential_decay, decay_rate=0.001)
 
EOF
)
