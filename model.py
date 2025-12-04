"""
Backpack Language Model Architecture
Based on the Backpack LM paper and nanoBackpackLM implementation
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint_sequential


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BackpackLM(nn.Module):
    """
    Backpack Language Model
    
    The key difference from standard transformers is that each word
    is represented as a weighted sum of sense vectors.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Sense vectors: each word has n_senses sense vectors
        self.n_senses = getattr(config, 'n_senses', 16)
        #self.sense_embeddings = nn.Embedding(config.vocab_size, config.n_embd * self.n_senses)


        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        self.sense_layer = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd * self.n_senses)
        )


        
        # Sense predictor: predicts weights for each sense
        self.sense_predictor = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, self.n_senses)
        )
        
        # Position embeddings
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Note: Weight tying with sense embeddings is more complex due to different shapes
        # We'll keep them separate for now

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, chunk_size=32):
        B, T = idx.size()
        
        # Get sense embeddings for each token: (B, T, n_senses * n_embd)
        #sense_embs = self.sense_embeddings(idx)  # (B, T, n_senses * n_embd)
        #sense_embs = sense_embs.view(B, T, self.n_senses, self.config.n_embd)

        token_embs = self.token_embedding(idx)            # (B, T, n_embd)

        # Get position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_embeddings(pos)  # (T, n_embd)
        
        # Use position embeddings as initial context to predict sense weights
        context = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, T, n_embd)
        sense_weights = self.sense_predictor(context)  # (B, T, n_senses)
        sense_weights = F.softmax(sense_weights, dim=-1)  # (B, T, n_senses)

        x_chunks=[]
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            token_chunk = token_embs[:, start:end, :]             # (B, chunk, n_embd)
            weights_chunk = sense_weights[:, start:end, :]        # (B, chunk, n_senses)
    
            # Compute sense vectors on-the-fly
            sense_embs_chunk = self.sense_layer(token_chunk)      # (B, chunk, n_embd * n_senses)
            sense_embs_chunk = sense_embs_chunk.view(B, end-start, self.n_senses, self.config.n_embd)
    
            # Weighted sum
            x_chunk = torch.einsum('btsd,bts->btd', sense_embs_chunk, weights_chunk)
            x_chunks.append(x_chunk)
        x = torch.cat(x_chunks, dim=1)
        
        # Weighted sum of sense vectors
        # sense_embs: (B, T, n_senses, n_embd)
        # sense_weights: (B, T, n_senses)
        #x = torch.einsum('btsd,bts->btd', sense_embs, sense_weights)  # (B, T, n_embd)
        
        # Add position embeddings, dropout
        x = x + pos_emb.unsqueeze(0)
        x = self.drop(x)
        
        segments = 2
        x = checkpoint_sequential(self.blocks, segments, x, use_reentrant=False)

        x = self.ln_f(x)
        
        if targets is not None:
            # Compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: return logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def get_sense_vectors(self, idx):
        """Extract sense vectors for given token indices"""
        B, T = idx.size()
        sense_embs = self.sense_embeddings(idx)  # (B, T, n_senses * n_embd)
        sense_embs = sense_embs.view(B, T, self.n_senses, self.config.n_embd)
        return sense_embs

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens given a context.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.config.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Apply temperature
            logits = logits / temperature
            # Optionally crop logits to top-k
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs.squeeze(1), num_samples=1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class StandardTransformerLM(nn.Module):
    """
    Standard Transformer Language Model (baseline)
    
    This is identical to BackpackLM but without sense vectors.
    Uses regular token embeddings instead of sense embeddings.
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # Regular token embeddings (instead of sense embeddings)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Position embeddings
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks (same as Backpack)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params/1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()

        # Get token embeddings (regular embeddings, not sense vectors)
        token_embs = self.token_embeddings(idx)  # (B, T, n_embd)
        
        # Get position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_embeddings(pos)  # (T, n_embd)
        
        # Add position embeddings to token embeddings
        x = token_embs + pos_emb.unsqueeze(0)  # (B, T, n_embd)
        x = self.drop(x)
        
        # Apply transformer blocks (same as Backpack)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        if targets is not None:
            # Compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: return logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Same optimizer configuration as Backpack
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens given a context.
        Same as Backpack generate method.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.config.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Apply temperature
            logits = logits / temperature
            # Optionally crop logits to top-k
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs.squeeze(1), num_samples=1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

