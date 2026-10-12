# C++ Code Completion Transformer

A decoder-only Transformer language model implemented from scratch with PyTorch and trained on a C++ source-code dataset. The model learns the structure and syntax of C++ from token sequences and can generate continuations for partial C++ prompts.

Given a prompt such as:

```cpp
int factorial(int n) {
```

the model predicts the next tokens one at a time and can produce a continuation such as a function body.

## Features

- Transformer language model implemented directly in Python with PyTorch.
- Causal self-attention so each position can attend only to earlier tokens.
- Token and positional embeddings.
- Multi-head self-attention blocks with feed-forward layers, residual connections, and normalization.
- Autoregressive next-token prediction.
- Training on a C++ source-code corpus.
- Text generation from an interactive or programmatic prompt.
- Configurable context length, embedding size, number of attention heads, number of layers, batch size, learning rate, and generation length.

## How It Works

The project uses the standard decoder-only language-modeling workflow:

```text
C++ source files
      |
      v
Tokenizer and vocabulary
      |
      v
Token IDs and training sequences
      |
      v
Transformer decoder
      |
      v
Next-token logits
      |
      v
Sampling / decoding
      |
      v
Generated C++ continuation
```

For every training sequence, the input and target are shifted by one token:

```text
Input:  #include <iostream> int main ( ) {
Target: include <iostream> int main ( ) { return
```

The model minimizes cross-entropy loss between its predicted distribution and the target next token. During generation, the predicted token is appended to the context and fed back into the model repeatedly.

## Transformer Architecture

The model is a causal decoder-only Transformer. A typical forward pass is:

```text
Token IDs
  -> Token embedding + positional embedding
  -> Transformer block 1
  -> Transformer block 2
  -> ...
  -> Transformer block N
  -> Final normalization
  -> Linear language-modeling head
  -> Vocabulary logits
```

Each Transformer block contains:

1. Masked multi-head self-attention.
2. A residual connection and normalization.
3. A position-wise feed-forward network.
4. A second residual connection and normalization.

The causal mask prevents the model from using future tokens while predicting the current next token. This makes the training objective consistent with autoregressive completion at inference time.

## Dataset

The model is trained on a C++ source-code dataset. The corpus should contain C++ files or text examples such as:

- `.cpp` implementation files
- `.h` and `.hpp` header files
- Classes, functions, templates, and namespaces
- Standard-library usage
- Control flow, memory management, and object-oriented code

Before training, the corpus is converted into token IDs using the project's tokenizer. The tokenizer must be used consistently during training and generation so that the model can decode generated IDs back into readable C++.

Recommended dataset preparation:

1. Collect C++ source files that you are permitted to use.
2. Remove generated binaries, build directories, and unrelated documentation.
3. Concatenate or independently tokenize the source files.
4. Split the token stream into training and validation portions.
5. Create fixed-length input-target windows using the configured context length.

Do not commit private, confidential, or license-restricted source code to the repository.

## Training Objective

For a token sequence $x_1, x_2, \ldots, x_T$, the model predicts:

$$
P(x_{t+1} \mid x_1, x_2, \ldots, x_t)
$$

The training loss is the average cross-entropy over all positions in a batch:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log P(x_{t+1} \mid x_1, \ldots, x_t)
$$

The optimizer updates the model parameters to reduce this loss over the C++ corpus.

## Requirements

Install the dependencies from the project's requirements file when available. The core runtime requires:

- Python 3.10 or newer
- PyTorch
- A tokenizer implementation used by the training and generation scripts
- NumPy, if used by the data pipeline

Example environment setup:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy
```

For CUDA training, install the PyTorch build matching the installed CUDA runtime from the official PyTorch installation instructions.

## Project Structure

Adapt the paths below to the filenames in the implementation:

```text
.
├── data/
│   ├── raw/                 C++ source corpus
│   └── processed/           Tokenized or prepared training data
├── checkpoints/             Saved model weights and optimizer state
├── src/
│   ├── model.py              Transformer architecture
│   ├── tokenizer.py          Token-to-ID and ID-to-token conversion
│   ├── dataset.py            Training and validation batches
│   ├── train.py              Training loop and checkpointing
│   └── generate.py           Autoregressive C++ completion
├── requirements.txt          Python dependencies
└── README.md                 Project documentation
```

## Training

Run the project's training entry point from the repository root. A typical command is:

```powershell
python train.py
```

or, when the training module is under `src/`:

```powershell
python -m src.train
```

The training script should:

- Load or prepare the C++ token dataset.
- Construct the Transformer and move it to the selected device.
- Create shifted input-target batches.
- Run forward and backward passes.
- Report training and validation loss.
- Save the best or latest checkpoint.

Useful configuration values to expose in the training script include:

```text
batch_size       Number of sequences per batch
block_size       Maximum context length
n_embd           Token embedding size
n_head           Number of attention heads
n_layer          Number of Transformer blocks
learning_rate    Optimizer learning rate
max_iters        Number of training iterations
eval_interval    Validation frequency
dropout          Dropout probability
device           cpu or cuda
```

Training from scratch can be computationally expensive. Start with a small model and a short run to verify the complete data, loss, checkpoint, and generation pipeline before increasing the configuration.

## Generate C++ Completions

After training, load the saved checkpoint and provide a partial C++ prompt:

```powershell
python generate.py --checkpoint checkpoints/model.pt --prompt "int main() {" --max-new-tokens 100
```

If the project uses a module entry point:

```powershell
python -m src.generate --checkpoint checkpoints/model.pt --prompt "class Vector {" --max-new-tokens 120
```

The generation process is:

1. Encode the prompt into token IDs.
2. Keep only the most recent context window when the prompt exceeds the model limit.
3. Run the Transformer in evaluation mode.
4. Convert the final logits into a probability distribution.
5. Select the next token using the configured decoding strategy.
6. Append the token and repeat until the requested length or an end token is reached.
7. Decode the resulting token IDs into C++ text.

Common decoding controls:

- `temperature`: lower values make output more deterministic; higher values increase variation.
- `top_k`: restricts sampling to the most likely `k` tokens.
- `top_p`: samples from the smallest probability set whose cumulative probability reaches `p`.
- `max_new_tokens`: maximum number of tokens generated after the prompt.

For a reproducible completion, set the random seed and use a deterministic decoding option such as greedy decoding or a low temperature.

## Example Prompts

```cpp
#include <iostream>

int main() {
```

```cpp
template <typename T>
class Stack {
public:
```

```cpp
std::vector<int> sort_values(std::vector<int> values) {
```

```cpp
for (int i = 0; i < n; ++i) {
```

The quality of a completion depends on whether the prompt style, formatting, libraries, and programming patterns are represented in the training corpus.

## Checkpoints

A checkpoint should contain at least the model parameters. For reliable training resumption, it should also contain:

- Optimizer state.
- Current iteration or epoch.
- Training and validation loss history.
- Model configuration.
- Tokenizer vocabulary or tokenizer configuration.
- Random-number-generator state when exact resumption is required.

The tokenizer vocabulary is part of the model contract. A checkpoint cannot reliably decode text unless it is paired with the same token-to-ID mapping used during training.

## Evaluation

Recommended evaluation checks include:

- Training and validation cross-entropy loss.
- Perplexity, computed as $\exp(\text{loss})$ when the loss is an average natural-log cross-entropy.
- Completion quality on held-out C++ snippets.
- Syntax validity of generated code.
- Whether generated functions, braces, and templates terminate correctly.
- Repetition, truncation, and malformed-token behavior.

Generated code should be treated as a suggestion. It must be reviewed, compiled, tested, and checked for security issues before use in a real application.

## Troubleshooting

### The loss does not decrease

- Confirm that inputs and targets are shifted by one token.
- Confirm that the causal mask has the correct shape and orientation.
- Check that token IDs are within the vocabulary range.
- Verify that the learning rate is appropriate for the model size.
- Run a tiny overfitting test on a single batch.

### Generated output is repetitive

- Check for an overly high or low learning rate.
- Increase the training corpus or improve its variety.
- Try a modest temperature or top-k sampling.
- Verify that the model is switched to evaluation mode during generation.

### Generated output is cut off

- Increase `max_new_tokens`.
- Check the configured context length.
- Confirm that the generation loop stops only on the intended end token.
- Remember that a fixed context window limits how much earlier code the model can use.

### CUDA is unavailable

The project can run on CPU for debugging and small experiments. For GPU training, verify the PyTorch installation, CUDA runtime compatibility, and the selected device configuration.

## Limitations

- The model predicts tokens from learned patterns; it does not formally understand or verify C++ semantics.
- It may generate code that does not compile, uses unavailable APIs, or contains logical errors.
- The model can reproduce biases, defects, or insecure patterns found in its training data.
- Long-range dependencies are limited by the context window.
- Quality depends strongly on corpus size, cleanliness, tokenization, and training duration.

## Responsible Use

Review generated code before compiling or deploying it. Do not execute untrusted generated code without appropriate isolation. Respect the licenses and usage terms of all C++ source code used to create the training corpus.

## License

Add the project's license here. If no license has been selected, state that explicitly before redistributing the repository or model weights.
