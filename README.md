
## 🔧 1. Biquad as a Neural Network

The biquad recurrence:

$$
y[n] = b_0 x[n] + b_1 x[n-1] + b_2 x[n-2] - a_1 y[n-1] - a_2 y[n-2]
$$

looks like a **2-tap input convolution + 2-tap recurrent connection**.
That is exactly what an **RNN cell** is:

* Inputs: \$x\[n], x\[n-1], x\[n-2]\$
* Hidden state: \$y\[n-1], y\[n-2]\$
* Weights: \${b\_0, b\_1, b\_2, a\_1, a\_2}\$

So a biquad is essentially a **2nd-order linear RNN** with linear activations.

---

## ⚡ 2. Neuralization

We can turn it into a **differentiable layer** in a NN:

* **Learnable parameters**: cutoff frequency, Q, and gain (mapped to biquad coeffs).
* **Forward pass**: apply recurrence as filtering.
* **Backward pass**: handled automatically by autograd → the NN can learn biquad parameters from data.

---

## 🧩 3. PyTorch Module Example

```python
import torch
import torch.nn as nn

class BiquadLayer(nn.Module):
    def __init__(self, fs=44100):
        super().__init__()
        self.f0 = nn.Parameter(torch.tensor(1000.0))  # center freq (Hz)
        self.Q  = nn.Parameter(torch.tensor(0.707))   # quality factor
        self.gain = nn.Parameter(torch.tensor(0.0))   # gain (dB)
        self.fs = fs

    def forward(self, x):
        # param -> coeffs
        w0 = 2 * torch.pi * self.f0 / self.fs
        alpha = torch.sin(w0) / (2 * self.Q)

        b0 = (1 - torch.cos(w0)) / 2
        b1 = 1 - torch.cos(w0)
        b2 = (1 - torch.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * torch.cos(w0)
        a2 = 1 - alpha

        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0

        # init states
        y = torch.zeros_like(x)
        x1 = x2 = y1 = y2 = 0.0

        # recurrent forward loop
        for n in range(len(x)):
            y[n] = b0*x[n] + b1*x1 + b2*x2 - a1*y1 - a2*y2
            x2, x1 = x1, x[n]
            y2, y1 = y1, y[n]

        return y
```

Now the **biquad is a trainable NN layer**. Gradients flow into `f0`, `Q`, and `gain`.

---

## 🚀 4. Going Beyond Linear Biquads

Because it’s now inside a NN:

* Replace linear recurrence with **nonlinear activations** → nonlinear filters.
* Stack multiple biquad layers → **learnable parametric EQ**.
* Mix with conv layers, attention, or transformers → **differentiable DSP hybrids**.
* Condition on images, text, or MIDI → **multimodal control filters**.

---

✅ So yes — the biquad **is a neural network already** (a linear 2nd-order RNN). And if we let its parameters be learned from data, it becomes a fully trainable NN layer.

Would you like me to extend this into a **bank of differentiable biquad layers** (like a trainable graphic EQ inside a deep model)?
