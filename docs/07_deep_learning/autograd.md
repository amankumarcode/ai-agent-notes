# Automatic Differentiation (Autograd)

**Updated:** 2025-07-26

## TL;DR

* **Goal:** Get gradients *automatically*—no hand-derived math required
* **Core idea:** Record every elementary operation in a *computational graph* during the forward pass, then apply the chain rule backward
* **Reverse-mode autograd** (used for deep networks) computes all parameter gradients with roughly the cost of one extra forward pass
* **Modern frameworks** like PyTorch, TensorFlow 2, and JAX wrap tensors so this recording happens transparently

---

## Why Do We Need Autograd?

> Training = tweaking millions of weights so the loss gets smaller.

Hand-coding $\frac{\partial \mathcal{L}}{\partial w}$ for each layer is error-prone and impossible at GPT scale. Autograd gives us those derivatives "for free," letting us focus on model architecture and ideas rather than calculus implementation.

**The Manual Alternative Would Be:**
- Deriving gradients analytically for every layer type
- Implementing backward passes for custom operations
- Debugging gradient computation errors
- Maintaining consistency as architectures evolve

**Autograd Eliminates All This:** Define the forward computation, get gradients automatically.

---

## The Core Recipe (Reverse-Mode)

The fundamental algorithm behind automatic differentiation:

1. **Forward pass** – compute output and *silently* build a computational graph of operations
2. **Seed gradient at the loss** – set $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1$
3. **Walk graph backward** – apply the chain rule to every node, caching $\frac{\partial \mathcal{L}}{\partial x}$ for its inputs
4. **Store gradients on parameters** – the optimizer reads these and updates weights

**Computational Efficiency:** Because each edge is visited once forward and once backward, runtime is approximately 2× a forward pass, regardless of the number of parameters.

---

## Minimal PyTorch Demo

```python
import torch

# Define tensors with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
W = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# Forward pass: graph recorded automatically
y_hat = W * x + b        # y_hat = 3*2 + 1 = 7
loss = (y_hat - 7)**2    # loss = (7-7)^2 = 0

# Backward pass: autograd magic
loss.backward()          # Fills .grad fields automatically

# Access computed gradients
print(f"W.grad: {W.grad}")  # tensor([0.]) since loss is already minimized
print(f"b.grad: {b.grad}")  # tensor([0.])

# For a more interesting example with non-zero gradients:
target = torch.tensor([5.0])
loss2 = (y_hat - target)**2  # loss = (7-5)^2 = 4

# Clear previous gradients
W.grad = None
b.grad = None

loss2.backward()
print(f"W.grad: {W.grad}")  # tensor([8.]) = 2*(7-5)*2
print(f"b.grad: {b.grad}")  # tensor([4.]) = 2*(7-5)*1
```

**Key Point:** `requires_grad=True` tells PyTorch to wrap each tensor so that operations get tracked in the computational graph.

---

## How Recording Works (Conceptual)

```mermaid
graph LR
    X[x<br/>requires_grad=True] --> M[multiply<br/>W*x]
    W[W<br/>requires_grad=True] --> M
    M --> A[add<br/>result + b]
    b[b<br/>requires_grad=True] --> A
    A --> S[square<br/>(result - target)²]
    target[target] --> S
    S --> L[loss<br/>scalar value]
    
    style X fill:#e1f5fe
    style W fill:#e1f5fe
    style b fill:#e1f5fe
    style L fill:#ffcdd2
```

**Graph Node Structure:**
Each node in the computational graph stores:
* **Forward value** computed during the forward pass
* **Backward function** to compute local gradients given upstream gradient
* **Input references** to propagate gradients backward

**Gradient Flow:**
During `backward()`, gradients flow from **loss** → **square** → **add** → **multiply** → **parameters**.

---

## Forward-Mode vs. Reverse-Mode

| Aspect | Forward-Mode | Reverse-Mode (Backprop) |
|--------|--------------|-------------------------|
| **Best for** | Few inputs, many outputs | Many inputs, few outputs |
| **Complexity** | $O(\text{num\_inputs})$ | $O(\text{num\_outputs})$ |
| **Memory** | Low | Higher (stores computation graph) |
| **Use case** | Jacobian-vector products | Neural network training |
| **Frameworks** | JAX `jvp`, PyTorch `forward_ad` | PyTorch, TensorFlow default |

**Why Reverse-Mode for Deep Learning:**
Neural networks typically have millions of parameters (inputs) but scalar losses (one output), making reverse-mode dramatically more efficient.

---

## Advanced Features and Patterns

### Higher-Order Gradients

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**3  # y = 8

# First-order gradient
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = {grad1}")  # 3*x^2 = 12

# Second-order gradient (gradient of gradient)
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"d²y/dx² = {grad2}")  # 6*x = 12
```

### Gradient Accumulation

```python
# Useful for large batch sizes that don't fit in memory
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

optimizer.zero_grad()
for mini_batch in large_dataset:
    loss = compute_loss(model(mini_batch))
    loss.backward()  # Accumulates gradients

optimizer.step()  # Update with accumulated gradients
```

### Selective Gradient Computation

```python
# Freeze certain parameters
for param in model.backbone.parameters():
    param.requires_grad = False

# Or temporarily disable gradients
with torch.no_grad():
    predictions = model(validation_data)  # No graph building
```

---

## Memory Management and Optimization

### Gradient Checkpointing

```python
import torch.utils.checkpoint as checkpoint

class MemoryEfficientBlock(torch.nn.Module):
    def forward(self, x):
        # Trade computation for memory
        return checkpoint.checkpoint(self._forward, x)
    
    def _forward(self, x):
        # Expensive computation here
        return expensive_function(x)
```

### Detaching from Computation Graph

```python
# Break gradient flow when needed
x = torch.randn(10, requires_grad=True)
y = expensive_computation(x)

# Use y's value but don't backprop through expensive_computation
z = some_function(y.detach())
loss = (z - target).sum()
loss.backward()  # Only backprops through some_function
```

---

## Common Questions and Answers

**Q: Why not use numerical finite differences for gradients?**  
**A:** Numerical methods require one forward pass per parameter (millions for large models), are computationally expensive, and suffer from numerical precision issues. Autograd computes exact gradients efficiently.

**Q: Does autograd store the whole computational graph in memory?**  
**A:** Yes, but you can manage memory usage with `torch.no_grad()`, `detach()`, gradient checkpointing, or by clearing graphs with `loss.backward(); optimizer.step(); optimizer.zero_grad()`.

**Q: When would I want forward-mode instead of reverse-mode?**  
**A:** Forward-mode is efficient for computing Jacobian-vector products when you have few inputs and many outputs. Examples include sensitivity analysis, uncertainty propagation, or computing directional derivatives.

**Q: Can I modify the computational graph during runtime?**  
**A:** Yes! PyTorch uses dynamic computational graphs, allowing conditional logic, loops, and runtime-dependent architectures. This enables flexible model designs and debugging.

**Q: How do I debug gradient computation?**  
**A:** Use gradient checking with finite differences, inspect intermediate gradients, use `torch.autograd.gradcheck()`, or visualize the computational graph with tools like `torch.fx` or `torchviz`.

---

## Common Pitfalls and Solutions

### 1. In-Place Operations

**Problem:**
```python
x = torch.tensor([1.0], requires_grad=True)
x += 1  # In-place operation can break autograd
```

**Solution:**
```python
x = torch.tensor([1.0], requires_grad=True)
x = x + 1  # Create new tensor, preserves gradient flow
```

### 2. Missing `requires_grad`

**Problem:**
```python
x = torch.tensor([1.0])  # Missing requires_grad=True
y = x * 2
y.backward()  # Error: no gradients to compute
```

**Solution:**
```python
x = torch.tensor([1.0], requires_grad=True)
# Or: x.requires_grad_(True)  # In-place modification
```

### 3. Memory Leaks in Training Loops

**Problem:**
```python
for batch in dataloader:
    loss = compute_loss(batch)
    losses.append(loss)  # Keeps entire computation graph in memory!
```

**Solution:**
```python
for batch in dataloader:
    loss = compute_loss(batch)
    losses.append(loss.item())  # Store only the scalar value
```

### 4. Gradients Not Zeroed

**Problem:**
```python
for epoch in range(num_epochs):
    loss = compute_loss()
    loss.backward()  # Gradients accumulate across epochs!
    optimizer.step()
```

**Solution:**
```python
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear gradients
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

---

## Framework Comparison

### PyTorch
```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # Dynamic graph, imperative style
```

### JAX
```python
import jax.numpy as jnp
from jax import grad

def f(x):
    return x**2

df_dx = grad(f)
print(df_dx(1.0))  # Functional programming style
```

### TensorFlow 2.x
```python
import tensorflow as tf

x = tf.Variable([1.0])
with tf.GradientTape() as tape:
    y = x**2
grad = tape.gradient(y, x)
print(grad)  # Explicit gradient tape
```

---

## Best Practices

### 1. Memory Management
- Use `torch.no_grad()` for inference
- Clear gradients regularly with `optimizer.zero_grad()`
- Consider gradient checkpointing for very deep models
- Monitor GPU memory usage during development

### 2. Numerical Stability
- Use appropriate data types (float32 vs float64)
- Implement gradient clipping for unstable training
- Check for NaN gradients in training loops
- Use numerically stable implementations of common operations

### 3. Performance Optimization
- Minimize Python loops in favor of vectorized operations
- Use `torch.jit.script` for performance-critical code
- Profile gradient computation with PyTorch profiler
- Consider mixed precision training for large models

### 4. Debugging and Validation
- Implement gradient checking for custom operations
- Use `register_hook` to inspect intermediate gradients
- Validate gradients with simple test cases
- Use deterministic operations during debugging

---

## Summary

Automatic differentiation is the cornerstone that makes modern deep learning practical. By automatically computing exact gradients through the chain rule, autograd systems enable:

- **Rapid prototyping** of new architectures without manual gradient derivation
- **Reliable gradients** free from human calculation errors
- **Scalable training** of models with millions to billions of parameters
- **Advanced techniques** like meta-learning and differentiable programming

**Key Takeaways:**
1. Autograd builds computational graphs during forward passes
2. Reverse-mode is optimal for the many-parameter, few-output case of neural networks
3. Modern frameworks handle the complexity while providing fine-grained control
4. Understanding autograd principles helps debug training issues and optimize performance

The transition from manual gradient computation to automatic differentiation represents one of the most significant advances in making deep learning accessible and practical for complex, real-world applications.

---

