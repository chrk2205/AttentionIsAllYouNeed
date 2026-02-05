import torch
import torch.nn as nn



class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, max_len : int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.positional_encoding = nn.Parameter(torch.zeros(d_model, max_len), requires_grad=False)

        for pos in range(d_model):
            for i in range(max_len):
                div_term =  (10000 ** (i / d_model))
                if i % 2 == 0:
                    self.positional_encoding[pos, i] = torch.sin(pos / div_term)
                else:
                    self.positional_encoding[pos, i] = torch.cos(pos / div_term)

    def forward(self, x):
        x = x + self.positional_encoding.unsqueeze(0)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):  # (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):  # (batch_size, seq_len, d_model)
        return self.net(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model : int , heads : int, dropout: float):
        self.dropout = nn.Dropout(dropout)

        assert d_model % heads
        self.d_model = d_model
        self.heads = heads

        self.d_k = d_model / heads

        self.w_q = nn.LazyLinear(d_model, bias = False) # in_features is also d_model
        self.w_k = nn.LazyLinear(d_model, bias = False) # in_features is also d_model
        self.w_v = nn.LazyLinear(d_model, bias = False) # in_features is also d_model
        self.w_o = nn.LazyLinear(d_model, bias = False) # in_features is also d_model

    

    def forward(self, q, k = None, v = None, mask = None):
        if k is None: k = q
        if v is None: v = q
        
        batch_size, query_len, _ = q.size()
        key_len = k.size(1)

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        scores = ( Q @ K.T / torch.sqrt(self.d_k) )

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)


        attention = nn.softmax(scores)
        attention = self.dropout(attention)
        out = attention @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        return self.w_o(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model : int, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model)
        self.feed_forward   = FeedForward(d_model, d_ff, dropout)
        self.layer_norm_1    = LayerNorm(d_model)
        self.layer_norm_2    = LayerNorm(d_model)

    def forward(self, encoder_input, encoder_mask):
        encoder_embeddings = self.self_attention(encoder_input, encoder_input, encoder_input, encoder_mask)
        encoder_embeddings_1 = self.layer_norm_1(encoder_input + encoder_embeddings)
        encoder_embeddings = self.feed_forward(encoder_embeddings_1)
        encoder_embeddings = self.layer_norm_2(encoder_embeddings_1 + encoder_embeddings)

        return encoder_embeddings



class Encoder(nn.Module):
    def __init__(self, vocab_size : int, d_model : int, num_layers : int = 5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model) for _ in range(num_layers)])
        
    def forward(self, encoder_input, encoder_mask):
        encoder_embeddings = self.embedding(encoder_input)
        encoder_embeddings = self.positional_encoding(encoder_embeddings)   
        pass

class DecoderLayer(nn.Module):
    pass


class Transformer(nn.Module):
    pass
