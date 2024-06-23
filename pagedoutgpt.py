import torch, torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.W_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, embedding_dim, bias=False)
    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        attn_scores = (Q @ K.transpose(1, 2)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        mask = torch.triu(torch.ones((1, Q.size(1), Q.size(1))), diagonal=1).to(x.device).masked_fill_(torch.ones((1, Q.size(1), Q.size(1))) == 1, float("-inf"))
        return torch.nn.functional.softmax(attn_scores + mask, dim=-1) @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, embeddings_dim, heads_count):
        super().__init__()
        self.single_head_size = embeddings_dim // heads_count
        self.heads = nn.ModuleList([AttentionHead(self.single_head_size) for _ in range(heads_count)])
        self.W_O = nn.Linear(embeddings_dim, embeddings_dim, bias=False)
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), len(self.heads), self.single_head_size).transpose(1, 2)
        concatenated = torch.cat([head(x[:, i]) for i, head in enumerate(self.heads)], dim=-1).transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.W_O(concatenated)

class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, embedding_dim, n=10000):
        super().__init__()
        self.pos_encodings = self.precompute_encodings(sequence_length, embedding_dim, n).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(self, x):
        return x + self.pos_encodings[:x.size(1)]
    def precompute_encodings(self, seq_len, emb_dim, n):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(torch.log(torch.tensor(n, dtype=torch.float)) / emb_dim))
        pos_encodings = torch.zeros((seq_len, emb_dim))
        pos_encodings[:, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 1::2] = torch.cos(position * div_term)
        return pos_encodings

class TransformerBlock(nn.Module):
    def __init__(self, embeddings_dim, heads_count):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embeddings_dim)
        self.multi_head_attn = MultiHeadAttention(embeddings_dim, heads_count)
        self.dropout1 = nn.Dropout(p=0.1)
        self.layer_norm2 = nn.LayerNorm(embeddings_dim)
        self.linear1 = nn.Linear(embeddings_dim, embeddings_dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(embeddings_dim * 4, embeddings_dim)
        self.dropout2 = nn.Dropout(p=0.1)
    def forward(self, x):
        x = x + self.dropout1(self.multi_head_attn(self.layer_norm1(x)))
        return x + self.dropout2(self.linear2(self.gelu(self.linear1(self.layer_norm2(x)))))

class GPT(nn.Module):
    def __init__(self, vocab_size, emb_dim, ctx_window, heads_count, blocks_count):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(ctx_window, emb_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(emb_dim, heads_count) for _ in range(blocks_count)])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, vocab_size)
    def forward(self, x):
        x = self.dropout(self.embeddings(x) + self.pos_enc(x))
        for block in self.transformer_blocks: x = block(x)
        return self.linear(self.layer_norm(x))
