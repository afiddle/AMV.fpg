 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a//dev/null b/docs/ML/adam_utils.py
index 0000000000000000000000000000000000000000..4e18444ed580a128ae597ad89edb06b918212176 100644
--- a//dev/null
+++ b/docs/ML/adam_utils.py
@@ -0,0 +1,66 @@
+"""Utility module implementing the Adam optimization update."""
+
+import numpy as np
+
+
+def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
+                                beta1=0.9, beta2=0.999, epsilon=1e-8):
+    """Update parameters using the Adam optimization algorithm.
+
+    Parameters
+    ----------
+    parameters : dict
+        Dictionary containing parameters ``Wl`` and ``bl``.
+    grads : dict
+        Dictionary containing gradients ``dWl`` and ``dbl``.
+    v : dict
+        Dictionary storing the exponentially weighted averages of gradients.
+    s : dict
+        Dictionary storing the exponentially weighted averages of squared gradients.
+    t : int
+        Time step used for bias correction.
+    learning_rate : float, optional
+        Learning rate. Default is ``0.01``.
+    beta1 : float, optional
+        Exponential decay hyperparameter for the first moment estimates.
+    beta2 : float, optional
+        Exponential decay hyperparameter for the second moment estimates.
+    epsilon : float, optional
+        Small constant to prevent division by zero. Default is ``1e-8``.
+
+    Returns
+    -------
+    tuple
+        Updated ``parameters``, ``v``, ``s``, ``v_corrected`` and ``s_corrected``.
+    """
+
+    L = len(parameters) // 2
+    v_corrected = {}
+    s_corrected = {}
+
+    for l in range(1, L + 1):
+        # Moving average of the gradients
+        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
+        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
+
+        # Compute bias-corrected first moment estimate
+        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
+        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)
+
+        # Moving average of the squared gradients
+        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
+        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
+
+        # Compute bias-corrected second raw moment estimate
+        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
+        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)
+
+        # Update parameters
+        parameters["W" + str(l)] -= learning_rate * (
+            v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
+        )
+        parameters["b" + str(l)] -= learning_rate * (
+            v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)
+        )
+
+    return parameters, v, s, v_corrected, s_corrected
 
EOF
)
