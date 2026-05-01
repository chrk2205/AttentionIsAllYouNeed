import torch
import torch.nn as nn
from logging import getLogger
from torch.nn import functional as F
import math

logger = getLogger(__name__)

def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices (0, 2, 4...) and cos to odd indices (1, 3, 5...)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 3. Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]        
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
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0
        
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads  # Dimension per head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None: k = q
        if v is None: v = q
        batch_size = q.size(0)

        # 1. Linear projections and split into heads
        # Resulting shape: (batch, seq_len, heads, d_k)
        Q = self.w_q(q).view(batch_size, -1, self.heads, self.d_k)
        K = self.w_k(k).view(batch_size, -1, self.heads, self.d_k)
        V = self.w_v(v).view(batch_size, -1, self.heads, self.d_k)

        # 2. Transpose to get heads to the 2nd dimension
        # Shape: (batch, heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # Scores shape: (batch, heads, query_len, key_len)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 4. Combine heads
        # Shape: (batch, query_len, heads * d_k)
        out = (attention @ V).transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.d_model)

        return self.w_o(out)

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



    def forward(self, decoder_input, encoder_output, self_att_mask, cross_att_mask):
        # self attention 
        decoder_embeddings = self.self_attention(decoder_input, decoder_input, decoder_input, self_att_mask)
        decoder_embeddings_1 = self.layer_norm_1(decoder_input + decoder_embeddings)


        # cross attention
        # not sure what mask to use here??
        cross_embeddings = self.cross_attention(decoder_embeddings, encoder_output, encoder_output, cross_att_mask)
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


    def forward(self, q, k, self_att_mask, cross_att_mask):
        q = self.embedding(q)
        q = self.positional_encoding(q)

        for layer in self.decoder_layers:
            q = layer(q, k, self_att_mask, cross_att_mask)        
        return q




class Transformer(nn.Module):
    def __init__(self, en_vocab_size : int, de_vocab_size:int, d_model : int, d_ff : int, num_layers : int = 4, dropout : float = 0.0):
        super().__init__()
        self.encoder = Encoder(en_vocab_size, d_model, d_ff, num_layers, dropout)
        self.decoder = Decoder(de_vocab_size, d_model, d_ff, num_layers, dropout)
        self.linear = nn.Linear(d_model, de_vocab_size)

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.encoder(encoder_input, encoder_mask)
        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask, encoder_mask)
        return self.linear(decoder_output)

    def create_causal_mask(self, seq_len, device):
        # x has shape (batch_size, seq_len)
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()        
        return mask == 0

    # @torch.eval()
    def generate(self, encoder_input, encoder_mask, start_token, end_token, max_new_tokens=100):
        device = encoder_input.device
        batch_size = encoder_input.shape[0] # Get actual batch size
        
        # 1. Encode once
        encoder_output = self.encoder(encoder_input, encoder_mask)
        # 2. Initialize decoder input with start_token for the WHOLE batch
        # Shape: (batch_size, 1)
        decoder_input = torch.empty(batch_size, 1).fill_(start_token).type_as(encoder_input).to(device)

        for _ in range(max_new_tokens):
            # 3. Create the causal mask for the current length of decoder_input
            # This mask should be (seq_len, seq_len) or (batch, seq_len, seq_len)
            curr_len = decoder_input.size(1)
            target_mask = self.create_causal_mask(curr_len, device)
            # 4. Forward pass
            # NOTE: Most decoders need BOTH the target_mask AND the encoder_mask
            out = self.decoder(decoder_input, encoder_output, target_mask, encoder_mask)
            
            # Projection to vocab
            logits = self.linear(out)

            # 5. Get the last token
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 6. Append
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # 7. Stop condition (only works easily if batch_size == 1)
            # For multi-batch, you'd usually check if ALL sequences hit end_token
            # if batch_size == 1 and next_token.item() == end_token:
            #     break

        return decoder_input



if __name__ == "__main__":
    vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    # ADDED: heads=8 to Transformer init (see Fix 3)
    transformer = Transformer(vocab_size=len(vocab), d_model=512, d_ff=2048, num_layers=4, dropout=0.1)
    
    encoder_input = torch.randint(0, len(vocab), (2, 100))
    decoder_input = torch.randint(0, len(vocab), (2, 100))
    
    # FIXED: Use unsqueeze(1).unsqueeze(2) to make shape (2, 1, 1, 100)
    encoder_mask = (encoder_input != 0).unsqueeze(1).unsqueeze(2)
    dec_pad_mask = (decoder_input != 0).unsqueeze(1).unsqueeze(2)
    
    look_ahead_mask = causal_mask(100).unsqueeze(0).unsqueeze(0)
    decoder_mask = dec_pad_mask & look_ahead_mask    
    
    print(transformer(encoder_input, decoder_input, encoder_mask, decoder_mask).shape)
    print(transformer.generate(encoder_input, encoder_mask, 0, 2, 100).shape)
