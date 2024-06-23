import torch, torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.W_Q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_K = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_V = nn.Linear(emb_dim, emb_dim, bias=False)
    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        attn_scores = (Q @ K.transpose(1, 2)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        mask = torch.triu(torch.ones((1, Q.size(1), Q.size(1))), diagonal=1).to(x.device)
        mask = mask.masked_fill_(torch.ones((1, Q.size(1), Q.size(1))) == 1, float("-inf"))
        return torch.nn.functional.softmax(attn_scores + mask, dim=-1) @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads):
        super().__init__()
        assert emb_dim % heads == 0
        self.head_size = emb_dim // heads
        self.heads = nn.ModuleList([AttentionHead(self.head_size) for _ in range(heads)])
        self.W_O = nn.Linear(emb_dim, emb_dim, bias=False)
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), len(self.heads), self.head_size).transpose(1, 2)
        concat = torch.cat([head(x[:, i]) for i, head in enumerate(self.heads)], dim=-1).transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.W_O(concat)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, emb_dim, n=10000):
        super().__init__()
        self.pos_enc = self.precompute_enc(seq_len, emb_dim, n).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(self, x):
        return x + self.pos_enc[:x.size(1)]
    def precompute_enc(self, seq_len, emb_dim, n):
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(torch.log(torch.tensor(n, dtype=torch.float)) / emb_dim))
        pos_enc = torch.zeros((seq_len, emb_dim))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        return pos_enc

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, heads)
        self.drop1 = nn.Dropout(p=0.1)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.lin1 = nn.Linear(emb_dim, emb_dim * 4)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(emb_dim * 4, emb_dim)
        self.drop2 = nn.Dropout(p=0.1)
    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        return x + self.drop2(self.lin2(
            self.gelu(self.lin1(self.norm2(x)))))

class GPT(nn.Module):
    def __init__(self, vocab_size, emb_dim, ctx_win, heads, blocks):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(ctx_win, emb_dim)
        self.drop = nn.Dropout(p=0.1)
        self.blocks = nn.ModuleList([TransformerBlock(emb_dim, heads) for _ in range(blocks)])
        self.norm = nn.LayerNorm(emb_dim)
        self.lin = nn.Linear(emb_dim, vocab_size)
    def forward(self, x):
        x = self.drop(self.emb(x) + self.pos_enc(x))
        for block in self.blocks: x = block(x)
        return self.lin(self.norm(x))
