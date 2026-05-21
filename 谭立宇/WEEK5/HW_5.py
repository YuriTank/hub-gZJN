import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import docx
import sys

class CausalSelfAttention(nn.Module):
    # Causal (masked) multi-head self-attention mechanism
    # Used in decoder-only Transformers to ensure each token only attends to past tokens
    def __init__(self, embed_dim, num_heads, dropout):
        # embed_dim: total dimensionality of the input/output
        # num_heads: number of parallel attention heads
        # dropout: dropout probability applied to attention weights
        super().__init__()
        assert embed_dim % num_heads == 0
        # Verify that dimensionality is evenly divisible into heads
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Dimensionality per head: split the embedding evenly across heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        # Single projection for Q, K, V simultaneously (efficient fused operation)
        # Maps embed_dim -> 3*embed_dim, then split into Q, K, V        
        self.proj = nn.Linear(embed_dim, embed_dim)
        # Output projection that concatenates and projects back to embed_dim
        self.dropout = nn.Dropout(dropout)
        # Dropout applied after softmax (on attention weights)

    def forward(self, x, mask):
        # x: input tensor of shape (B, T, C) where B=batch, T=sequence length, C=embed_dim
        # mask: causal mask of shape (1, 1, max_seq_len, max_seq_len)
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        # Project input to QKV, then reshape to separate heads
        # Output shape: (B, T, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)
        # Split into separate Q, K, V tensors along dimension 2, each (B, T, num_heads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # Transpose to (B, num_heads, T, head_dim) for batched dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        # Compute scaled dot-product attention: (Q @ K^T) / sqrt(d_k)
        attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        # Apply causal mask: set positions that should not attend to -inf
        # mask[:, :, :T, :T] selects the relevant portion for current sequence length T
        attn = F.softmax(attn, dim=-1)
        # Softmax over last dimension (keys) to get attention probabilities
        attn = self.dropout(attn)
        # Apply dropout to attention weights (regularization)
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        # Weighted sum of values: output shape (B, num_heads, T, head_dim)
        return self.proj(out)
        # Apply output projection and return

class FeedForward(nn.Module):
    # Position-wise feed-forward network (MLP) applied to each token independently
    def __init__(self, embed_dim, ff_dim, dropout):
        # embed_dim: input/output dimensionality
        # ff_dim: hidden dimensionality (typically 4x embed_dim)
        # dropout: dropout probability applied after each linear layer
        super().__init__()
        self.net = nn.Sequential(
            
            nn.Linear(embed_dim, ff_dim),
            # First linear projection: expand to ff_dim
            nn.GELU(),
            # GELU activation (smooth version of ReLU, used in GPT variants)
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            # Second linear projection: contract back to embed_dim
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
class DecoderBlock(nn.Module):
    # A single Transformer decoder layer with pre-LayerNorm architecture
    # Consists of causal self-attention + feed-forward network, each with residual connections
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        # Pre-attention layer normalization
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        # Causal multi-head self-attention sub-layer
        self.ln2 = nn.LayerNorm(embed_dim)
        # Pre-FFN layer normalization
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        # Position-wise feed-forward sub-layer

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        # Pre-LN: apply LayerNorm before attention, then residual connection
        x = x + self.ff(self.ln2(x))
        # Pre-LN: apply LayerNorm before FFN, then residual connection
        return x
class CausalTransformerLM(nn.Module):
    # Decoder-only causal language model (similar to GPT)
    # Takes token indices, outputs logits over vocabulary
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4,
                 ff_dim=512, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        # Token embedding table: maps vocab indices to dense vectors
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        # Position embedding table: adds positional information to tokens
        self.dropout = nn.Dropout(dropout)
        # Dropout applied to sum of token + position embeddings
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        # Stack of decoder blocks (transformer layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        # Final layer normalization before the LM head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Language modeling head: projects from embed_dim back to vocab_size

        self.token_embed.weight = self.lm_head.weight
        # Weight tying: share weights between token embedding and LM head
        # This reduces parameter count and improves training
        self.register_buffer('causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))
        # Pre-computed causal attention mask (lower-triangular matrix)
        # Shape: (1, 1, max_seq_len, max_seq_len)

    def forward(self, x, targets=None):
        B, T = x.shape
        # x: input token indices of shape (B, T)
        # targets: optional ground-truth token indices for loss computation
        
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        # Create position indices (0, 1, ..., T-1) and broadcast across batch
        
        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.dropout(x)
        # Token embeddings + positional embeddings, then dropout
        for block in self.blocks:
            x = block(x, self.causal_mask)
        # Pass through each decoder block in sequence
        logits = self.lm_head(self.ln_f(x))
        # Apply final layer norm, then project to vocabulary logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Cross-entropy loss: flatten logits and targets over batch*time dimensions
        return logits, loss
    @torch.no_grad()

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # Autoregressive text generation without gradient tracking
        # idx: initial token indices of shape (1, T)
        # max_new_tokens: number of tokens to generate
        # temperature: higher = more random, lower = more deterministic
        # top_k: if set, only sample from top-k highest probability tokens
        self.eval()
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.causal_mask.shape[-1]:]
            # Truncate to max_seq_len if current sequence exceeds context window            
            logits, _ = self.forward(idx_cond)
            # Forward pass to get logits for next token
            logits = logits[:, -1, :] / temperature
            # Take only the last timestep's logits and apply temperature scaling
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float('-inf')
                # Keep only top-k values, set rest to -inf
            probs = F.softmax(logits, dim=-1)
            # Convert logits to probabilities via softmax
            next_token = torch.multinomial(probs, num_samples=1)
            # Sample next token from probability distribution
            idx = torch.cat((idx, next_token), dim=1)
            # Append generated token to sequence
        return idx

# Default training text (Biblical Genesis excerpt for demo)
TEXT = (
    "In the beginning God created the heaven and the earth. "
    "And the earth was without form and void and darkness was upon the face of the deep. "
    "And the Spirit of God moved upon the face of the waters. "
    "And God said Let there be light and there was light. "
    "And God saw the light that it was good and God divided the light from the darkness. "
    "And God called the light Day and the darkness he called Night. "
    "And the evening and the morning were the first day. "
    "And God said Let there be a firmament in the midst of the waters "
    "and let it divide the waters from the waters. "
    "And God made the firmament and divided the waters which were under the firmament "
    "from the waters which were above the firmament and it was so. "
    "And God called the firmament Heaven. "
    "And the evening and the morning were the second day. "
)
class CharTokenizer:
    # Character-level tokenizer: maps characters to/from integer IDs
    def __init__(self, text):
        chars = sorted(set(text))
        # Extract all unique characters and sort them
        self.stoi = {c: i for i, c in enumerate(chars)}
        # Character to index mapping
        self.itos = {i: c for i, c in enumerate(chars)}
        # Index to character mapping
        self.vocab_size = len(chars)
    def encode(self, s):
        return [self.stoi[c] for c in s]
        # Convert a string to a list of integer token IDs
    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)
        # Convert a list of integer token IDs back to a string

def read_text_from_docx(filepath):
    doc = docx.Document(filepath)
    # Read text content from a .docx file
    
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return '\n'.join(paragraphs)
    # Extract text from paragraphs, filtering out empty ones
    
def get_batch(data, batch_size, seq_len):
    # Sample a batch of random contiguous sequences from training data
    # data: 1D tensor of token IDs
    # batch_size: number of sequences per batch
    # seq_len: length of each sequence (context window)
    n = len(data)
    ix = torch.randint(0, n - seq_len, (batch_size,))
    # Random starting positions for each sequence in the batch
    
    x = torch.stack([data[i:i + seq_len] for i in ix])
    # Input sequences: data[i : i+seq_len]
    
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    # Target sequences (shifted by 1): data[i+1 : i+seq_len+1]
    
    return x, y

def train(docx_path=None):
    # Main training function
    if docx_path:
    # docx_path: optional path to .docx file with training text
        print(f"Reading text from: {docx_path}")
        text = read_text_from_docx(docx_path)
        print(f"Loaded {len(text)} characters\n")
        # Load text from external .docx file if provided
    else:
        # Fall back to hardcoded TEXT constant
        text = TEXT
    tokenizer = CharTokenizer(text)
    # Build character-level tokenizer from the text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    # Encode entire text into tensor of token IDs

    # Model hyperparameters (small config for fast demo training)
    vocab_size = tokenizer.vocab_size
    embed_dim = 64
    num_heads = 2
    num_layers = 3
    ff_dim = 128
    max_seq_len = 48
    dropout = 0.1

    # Training hyperparameters
    batch_size = 16
    epochs = 30
    lr = 0.001

    # Instantiate the model
    model = CausalTransformerLM(
        vocab_size, embed_dim, num_heads, num_layers,
        ff_dim, max_seq_len, dropout
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # AdamW optimizer (decoupled weight decay, standard for Transformers)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Text length: {len(data)} characters\n")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        steps = 0
        for _ in range(30):
        # Each epoch: 30 mini-batch gradient updates

            x, y = get_batch(data, batch_size, max_seq_len)
            # Sample a random batch of sequences

            logits, loss = model(x, y)
            # Forward pass: compute logits and loss

            optimizer.zero_grad()
            # Zero out accumulated gradients

            loss.backward()
            # Backward pass: compute gradients via backpropagation

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Gradient clipping: prevent exploding gradients by capping norm to 1.0

            # Update model parameters using computed gradients
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        avg_loss = epoch_loss / steps
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss: {avg_loss:.4f}")
    print("\nTraining complete!\n")
    # Generate text to demonstrate the trained model
    prompt = text[:10].strip()
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    output_ids = model.generate(input_ids, max_new_tokens=80, temperature=0.8, top_k=10)
    # Generate 80 new tokens with temperature 0.8 and top-k=10 sampling
    generated = tokenizer.decode(output_ids[0].tolist())
    print("=" * 50)
    print("Generation result:")
    print("=" * 50)
    print(generated)
    print("=" * 50)
if __name__ == "__main__":
    # Entry point: if a .docx path is provided as CLI argument, load that text
    # Otherwise use the default Bible verse text
    if len(sys.argv) > 1:
        train(docx_path=sys.argv[1])
    else:
        train()
