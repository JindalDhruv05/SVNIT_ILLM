# Detailed Guide: Custom Multinomial Logistic Regression

This document provides a detailed breakdown of the mathematical formulation, implementation details, and optimization mechanics behind the custom Multinomial (Multi-class) Logistic Regression classifier used in the Language Identification tasks (`lab1` and `lab2`).

---

## 1. Model Structure & Dimensions

In a classification task with $N$ samples, $F$ features (representing the vocabulary size), and $C$ target classes (representing the languages), the model is parameterized by:

*   **Input Features ($X$):** A 2D array of shape $(N, F)$ containing the TF-IDF feature vectors.
*   **Weights ($W$):** A 2D array of shape $(F, C)$, where each column $c$ corresponds to the feature weights for class $c$.
*   **Biases ($b$):** A 1D array of shape $(C,)$, representing the baseline probability offset for each class.

---

## 2. Mathematical Formulation

### 2.1 The Linear Step (Logits)
For each document, we first compute raw score indicators (logits) for each class by taking the dot product of the features and weights, then adding the bias:

$$Z = XW + b$$

*   Mathematically, for a single sample $x_i$ (a row vector of shape $1 \times F$):
    $$z_{i, c} = \sum_{j=1}^{F} x_{i, j} W_{j, c} + b_c$$
*   The resulting matrix $Z$ has a shape of $(N, C)$.

### 2.2 The Softmax Activation
To translate raw logits $Z$ into class probabilities that sum to 1, we apply the **Softmax function** across each row:

$$P(y_i = c \mid x_i) = \frac{e_{i, c}^z}{\sum_{k=1}^C e^{z_{i, k}}}$$

#### Numerical Stability Trick
In raw computation, computing $e^z$ can cause floating-point overflow errors if $z$ is large (e.g., $e^{709}$ is the limit for 64-bit floats). To prevent this, we subtract the maximum value of each row from its elements:

$$z'_{i, c} = z_{i, c} - \max_{j} (z_{i, j})$$
$$P(y_i = c \mid x_i) = \frac{e^{z'_{i, c}}}{\sum_{k=1}^C e^{z'_{i, k}}}$$

This operation is mathematically equivalent to the original softmax:

$$\frac{e^{z_{i, c} - m}}{\sum e^{z_{i, k} - m}} = \frac{e^{z_{i, c}} \cdot e^{-m}}{\sum e^{z_{i, k}} \cdot e^{-m}} = \frac{e^{z_{i, c}}}{\sum e^{z_{i, k}}}$$

However, it guarantees that the largest value exponentiated is $e^0 = 1$, ensuring **absolute numerical stability** without altering the probabilities.

---

## 3. Objective Loss Function

The objective of training is to minimize the **Categorical Cross-Entropy Loss** combined with **$L_2$ Regularization** (often called Weight Decay or Ridge penalty):

$$J(W, b) = -\frac{1}{N} \sum_{i=1}^N \log(P_{i, y_i}) + \frac{\lambda}{2} \|W\|_F^2$$

Where:
*   $P_{i, y_i}$ is the predicted probability of the correct class $y_i$ for sample $i$.
*   $\lambda$ (represented in code as `l2_reg`) is the regularization strength.
*   $\|W\|_F^2 = \sum_{j=1}^F \sum_{k=1}^C W_{j,k}^2$ is the squared Frobenius norm of the weights.

---

## 4. Backpropagation & Optimization

### 4.1 Gradients of the Loss
To update the parameters using Gradient Descent, we calculate the partial derivatives of the loss $J$ with respect to the weights $W$ and biases $b$. 

Let $Y$ be the one-hot encoded matrix of shape $(N, C)$ representing the true labels (where $Y_{i, c} = 1$ if the true label of sample $i$ is $c$, and $0$ otherwise).

1.  **Error delta ($dZ$):**
    $$dZ = \frac{\partial J}{\partial Z} = \frac{P - Y}{N}$$
    *Shape: $(N, C)$*

2.  **Weight Gradients ($dW$):**
    Using the chain rule, the gradient of the loss with respect to $W$ is:
    $$dW = \frac{\partial J}{\partial W} = X^T dZ + \lambda W$$
    *Shape: $(F, C)$*

3.  **Bias Gradients ($db$):**
    The gradient of the loss with respect to $b$ is:
    $$db = \frac{\partial J}{\partial b} = \sum_{i=1}^N dZ_i$$
    *Shape: $(C,)$*

### 4.2 Mini-batch Stochastic Gradient Descent (SGD)
Rather than computing gradients on the entire dataset at once (which is memory-heavy), we use **Mini-batch SGD**:
1.  **Shuffle:** At the start of each epoch, shuffle the training indices.
2.  **Batches:** Slice $X$ and $Y$ into small blocks of size $B$ (default is 256).
3.  **Update:** For each batch, calculate the gradients $dW_{\text{batch}}$ and $db_{\text{batch}}$ and adjust parameters using the learning rate $\alpha$:
    $$W \leftarrow W - \alpha \cdot dW_{\text{batch}}$$
    $$b \leftarrow b - \alpha \cdot db_{\text{batch}}$$

---

## 5. Weight Initialization

Proper weight initialization prevents signals from vanishing or exploding during the first forward pass. The class implements **Xavier Normal (Glorot) Initialization**:

$$W \sim \mathcal{N}\left(0, \sigma^2\right) \quad \text{where} \quad \sigma = \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$$

In our case:
*   $\text{fan\_in} = F$ (number of input features)
*   $\text{fan\_out} = C$ (number of classes)
*   Biases ($b$) are initialized to zero vectors.
