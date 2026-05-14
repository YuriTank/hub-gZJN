import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import math


class TransformerLayer(nn.Module):
    # Create a single Transformer layer with:
    # - Multi-Head Self-Attention
    # - Feed-Forward Network
    # - Layer Normalization and Residual Connections
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()
        
        # Initialize the Multi-Head Attention layer
        # d_model: dimension of the model (embedding size)
        # num_heads: number of attention heads
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True  # Input format: (batch, seq_len, d_model)
        )
        
        # d_model: input and output dimension
        # d_ff: hidden layer dimension (usually 2-4x d_model)
        # Feed-Forward Network (two fully connected layers)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization for the attention output
        # Add & Norm step 1: normalize after attention
        self.norm1 = nn.LayerNorm(d_model)
        
        # Layer normalization for the feed-forward output
        # Add & Norm step 2: normalize after FFN
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Save the input for residual connection
        # This is the "Add" part of "Add & Norm"
        residual = x
        
        # Step 1: Multi-Head Self-Attention
        # Compute attention scores and apply attention
        # The output shape is (batch, seq_len, d_model)
        attn_output, _ = self.self_attention(
            query=x,          # Query: the input to be transformed
            key=x,            # Key: same as input for self-attention
            value=x,          # Value: same as input for self-attention
            attn_mask=mask    # Optional mask for masking future positions
        )
        
        # Apply dropout to attention output
        attn_output = self.dropout(attn_output)
        
        # Add (residual connection) and Normalize
        # Combine attention output with original input
        x = self.norm1(residual + attn_output)
        
        # Save the input for the second residual connection
        residual = x
        
        # Step 2: Feed-Forward Network
        # Process through two linear layers with ReLU activation
        ff_output = self.feed_forward(x)
        
        # Add (residual connection) and Normalize
        x = self.norm2(residual + ff_output)
        
        return x


class SimpleTransformer(nn.Module):
    # Create a complete Transformer model with:
    # - Input embedding
    # - Positional encoding
    # - Multiple Transformer layers
    # - Output projection
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=5000, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        
        # Model dimension parameter
        self.d_model = d_model
        
        # Embedding layer to convert token indices to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding to inject position information
        # Sinusoidal encoding for even indices, cosine for odd indices
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Stack of Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)  # Repeat for each layer
        ])
        
        # Output projection layer (optional: projects to vocabulary size)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len, d_model):
        # Create sinusoidal positional encoding.
        # This encoding helps the model understand the position of each token.
        # Formula:
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        # Initialize an empty tensor for positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create a column vector of positions [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term: 10000^(2i/d_model)
        # This creates a decreasing frequency for higher dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension for broadcasting
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        # Register as a buffer (not a trainable parameter)
        return nn.Buffer(pe)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len)
        # x shape: (batch_size, seq_len)
        
        # Get the sequence length
        seq_len = x.size(1)
        
        # Convert token indices to embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding to the embeddings
        # Add positional encoding to the embeddings
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Apply dropout to embeddings
        x = self.dropout(x)
        
        # Pass through each Transformer layer
        for layer in self.layers:
            x = layer(x, mask)
        
        # Project to vocabulary size for output logits
        output = self.output_projection(x)
        
        return output


if __name__ == "__main__":
    # Example
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    VOCAB_SIZE = 10000    # Size of the vocabulary
    D_MODEL = 512        # Model dimension
    NUM_HEADS = 8         # Number of attention heads
    D_FF = 2048          # Feed-forward hidden dimension
    NUM_LAYERS = 6        # Number of Transformer layers
    BATCH_SIZE = 32      # Batch size
    SEQ_LEN = 128        # Sequence length
    
    # Create the model
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS
    )
    
    # Create sample input (batch of token indices)
    # Each number represents a token in the vocabulary
    input_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_tokens)
    
    # Output shape: (batch_size, seq_len, vocab_size)
    print(f"Input shape: {input_tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
