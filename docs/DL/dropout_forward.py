 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a//dev/null b/docs/ML/dropout_forward.py
index 0000000000000000000000000000000000000000..70a3e55559fd0bd507439b672b03dca51d31a37f 100644
--- a//dev/null
+++ b/docs/ML/dropout_forward.py
@@ -0,0 +1,62 @@
+"""Forward propagation with dropout for a 3-layer neural network."""
+
+import numpy as np
+
+
+def relu(Z):
+    """ReLU activation function."""
+    return np.maximum(0, Z)
+
+
+def sigmoid(Z):
+    """Sigmoid activation function."""
+    return 1 / (1 + np.exp(-Z))
+
+
+def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
+    """Implement forward propagation with dropout on a 3-layer network.
+
+    Parameters
+    ----------
+    X : ndarray
+        Input data of shape (2, number of examples).
+    parameters : dict
+        Dictionary containing weights and biases.
+    keep_prob : float, optional
+        Probability of keeping a neuron active during dropout.
+
+    Returns
+    -------
+    tuple
+        Output activation ``A3`` and a cache for backpropagation.
+    """
+
+    np.random.seed(1)
+
+    W1 = parameters["W1"]
+    b1 = parameters["b1"]
+    W2 = parameters["W2"]
+    b2 = parameters["b2"]
+    W3 = parameters["W3"]
+    b3 = parameters["b3"]
+
+    Z1 = np.dot(W1, X) + b1
+    A1 = relu(Z1)
+    D1 = np.random.rand(*A1.shape)
+    D1 = (D1 < keep_prob).astype(int)
+    A1 = A1 * D1
+    A1 = A1 / keep_prob
+
+    Z2 = np.dot(W2, A1) + b2
+    A2 = relu(Z2)
+    D2 = np.random.rand(*A2.shape)
+    D2 = (D2 < keep_prob).astype(int)
+    A2 = A2 * D2
+    A2 = A2 / keep_prob
+
+    Z3 = np.dot(W3, A2) + b3
+    A3 = sigmoid(Z3)
+
+    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
+
+    return A3, cache
 
EOF
)
