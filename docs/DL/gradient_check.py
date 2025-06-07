 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a//dev/null b/gradient_check.py
index 0000000000000000000000000000000000000000..8dc2fd19648b7dcaab49b49900678c885fdb5897 100644
--- a//dev/null
+++ b/gradient_check.py
@@ -0,0 +1,62 @@
+import numpy as np
+
+def forward_propagation(x, theta):
+    """Compute the cost for a simple linear function."""
+    J = theta * x
+    return J
+
+
+def backward_propagation(x, theta):
+    """Compute the derivative of the cost with respect to theta."""
+    dtheta = x
+    return dtheta
+
+
+def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
+    """Implement gradient checking for a simple function.
+
+    Parameters
+    ----------
+    x : float
+        Input value.
+    theta : float
+        Parameter of the function.
+    epsilon : float, optional
+        Small shift used for the numerical approximation of the gradient.
+    print_msg : bool, optional
+        Whether to print a diagnostic message.
+
+    Returns
+    -------
+    float
+        Relative difference between numerical and analytical gradients.
+    """
+
+    # Compute gradapprox using formula (1)
+    theta_plus = theta + epsilon
+    theta_minus = theta - epsilon
+    J_plus = forward_propagation(x, theta_plus)
+    J_minus = forward_propagation(x, theta_minus)
+    gradapprox = (J_plus - J_minus) / (2 * epsilon)
+
+    # Compute the gradient using backward propagation
+    grad = backward_propagation(x, theta)
+
+    # Compute the difference using formula (2)
+    numerator = np.linalg.norm(grad - gradapprox)
+    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
+    difference = numerator / denominator
+
+    if print_msg:
+        if difference > 2e-7:
+            print(
+                "\033[93mThere is a mistake in the backward propagation! difference =" \
+                f" {difference}\033[0m"
+            )
+        else:
+            print(
+                "\033[92mYour backward propagation works perfectly fine! difference =" \
+                f" {difference}\033[0m"
+            )
+
+    return difference
 
EOF
)
