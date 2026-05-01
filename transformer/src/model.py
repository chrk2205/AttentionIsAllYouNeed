import torch
import torch.nn as nn
from logging import getLogger
from torch.nn import functional as F

logger = getLogger(__name__)

def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, max_len : int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.positional_encoding = nn.Parameter(torch.zeros(d_model, max_len), requires_grad=False)

        for pos in range(d_model):
            for i in range(max_len):
                div_term =  torch.tensor(10000 ** (i / d_model))
                if i % 2 == 0:
                    self.positional_encoding[pos, i] = torch.sin(torch.tensor(pos) / div_term)
                else:
                    self.positional_encoding[pos, i] = torch.cos(torch.tensor(pos) / div_term)


    def forward(self, x):      
        pe = self.positional_encoding.t().unsqueeze(0)
        return x + pe


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
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads

        embed_size = d_model

        self.d_k = torch.tensor(d_model / heads)

        self.w_q = nn.LazyLinear(embed_size, bias = False) # in_features is also d_model
        self.w_k = nn.LazyLinear(embed_size, bias = False) # in_features is also d_model
        self.w_v = nn.LazyLinear(embed_size, bias = False) # in_features is also d_model
        self.w_o = nn.LazyLinear(embed_size, bias = False) # in_features is also d_model

    

    def forward(self, q, k = None, v = None, mask = None):
        if k is None: k = q
        if v is None: v = q

        batch_size, query_len, _ = q.size()
        key_len = k.size(1)

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
     
        scores = ( Q @ K.transpose(-2,-1) / torch.sqrt(self.d_k) )

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)


        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        out = attention @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        out = self.w_o(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, d_model : int, d_ff, heads, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward   = FeedForward(d_model, d_ff, dropout)
        self.layer_norm_1    = LayerNorm(d_model)
        self.layer_norm_2    = LayerNorm(d_model)

    def forward(self, encoder_input, encoder_mask):
        # self attention
        encoder_embeddings = self.self_attention(encoder_input, encoder_input, encoder_input, encoder_mask)
        encoder_embeddings_1 = self.layer_norm_1(encoder_input + encoder_embeddings)
        encoder_embeddings = self.feed_forward(encoder_embeddings_1)
        encoder_embeddings = self.layer_norm_2(encoder_embeddings_1 + encoder_embeddings)

        return encoder_embeddings



class Encoder(nn.Module):
    def __init__(self, vocab_size : int, d_model : int, d_ff : int, num_layers : int = 4, dropout : float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, d_ff, num_layers,  dropout) for _ in range(num_layers)])

        
    def forward(self, encoder_input, encoder_mask):
        # since this is self attention, i the encoder mask can be all 1ones
        x = self.embedding(encoder_input)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, encoder_mask)

        return  x



class DecoderBlock(nn.Module):
    def __init__(self, d_model : int, d_ff, heads, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.layer_norm_1    = LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model, heads, dropout)
        self.layer_norm_2    = LayerNorm(d_model)

        self.feed_forward   = FeedForward(d_model, d_ff, dropout)
        self.layer_norm_3    = LayerNorm(d_model)



    def forward(self, decoder_input, encoder_output, mask):
        # self attention 
        decoder_embeddings = self.self_attention(decoder_input, decoder_input, decoder_input, mask)
        decoder_embeddings_1 = self.layer_norm_1(decoder_input + decoder_embeddings)


        # cross attention
        cross_embeddings = self.cross_attention(decoder_embeddings, encoder_output, encoder_output, mask)
        cross_embeddings = self.layer_norm_2(decoder_embeddings_1 + cross_embeddings)


        cross_embeddings_1 = self.feed_forward(cross_embeddings)
        cross_embeddings_1 = self.layer_norm_3(cross_embeddings + cross_embeddings_1)

        return cross_embeddings_1



class Decoder(nn.Module):
    def __init__(self,vocab_size : int, d_model : int, d_ff : int, num_layers : int = 4, dropout : float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, d_ff, num_layers, dropout) for _ in range(num_layers)])


    def forward(self, q, k, mask):
        q = self.embedding(q)
        q = self.positional_encoding(q)

        for layer in self.decoder_layers:
                q = layer(q, k, mask)        
        return q




class Transformer(nn.Module):
    def __init__(self, vocab_size : int, d_model : int, d_ff : int, num_layers : int = 4, dropout : float = 0.0):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, d_ff, num_layers, dropout)
        self.decoder = Decoder(vocab_size, d_model, d_ff, num_layers, dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.encoder(encoder_input, encoder_mask)
        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask)
        return self.linear(decoder_output)




if __name__ == "__main__":
    vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    transformer = Transformer(vocab_size=len(vocab), d_model=512, d_ff=2048, num_layers=4, dropout=0.1)
    # print(transformer)
    encoder_input = torch.randint(0, len(vocab), (1, 100))
    decoder_input = torch.randint(0, len(vocab), (1, 100))
    encoder_mask = (encoder_input != torch.tensor(0)).unsqueeze(0).unsqueeze(0)
    decoder_mask = (decoder_input != torch.tensor(0)).unsqueeze(0).unsqueeze(0)
    decoder_mask = decoder_mask & causal_mask(100).unsqueeze(0)
    print(transformer(encoder_input, decoder_input, encoder_mask, decoder_mask).shape)
