import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

embed_size = 64 

class Head(nn.Module):

    def __init__(self, head_size, block_size=32, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size,  bias=False)
        self.query = nn.Linear(embed_size, head_size,  bias=False)
        self.value = nn.Linear(embed_size, head_size,  bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        
        v = self.value(x)
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        """
        i think the wei here should be close to this
                1,0,0,0,0
                1,1,0,0,0
                1,1,1,0,0
                1,1,1,1,0
                1,1,1,1,1
        It represent that the future tokens cannot talk with the previous tokens, which is the autoregresive decoder
        """
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, head_size=64, block_size=32, n_heads = 4, n_embed = 64, dropout = 0.0):
        super().__init__()
        self.sa_heads = nn.ModuleList([Head(head_size, block_size, dropout) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)


    def forward(self,x):
        x = torch.cat([h(x) for h in self.sa_heads], dim=-1)
        x = self.dropout(self.projection(x))
        return x 

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_heads):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiheadAttention(head_size=head_size, n_heads=n_heads)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size, n_embed = 64, block_size=32, n_heads = 4, n_layer = 4) -> None:
    super().__init__()
    self.block_size = block_size # max token size 
    self.token_embeddings_table = nn.Embedding(vocab_size, n_embed)
    self.position_tokens = nn.Embedding(block_size, n_embed)

    self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed) # final layer norm
    self.lm_head = nn.Linear(n_embed, vocab_size)


  def forward(self, idx):
    _, T = idx.shape
    tok_emb = self.token_embeddings_table(idx)
    pos_emb = self.position_tokens(torch.arange(T,device=tok_emb.device))

    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    x = self.lm_head(x)

    return x

  def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx