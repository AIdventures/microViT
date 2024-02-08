"""
ViT implementation in PyTorch.
References:
1) The official paper: https://arxiv.org/pdf/2010.11929.pdf
2) nanoGPT, by Karpathy: https://github.com/karpathy/nanoGPT
3) My personal blog post: https://aidventure.es/blog/vit
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Conv2d(
            config.in_channels, config.n_embd,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        self.n_patches = (config.img_size // config.patch_size) ** 2

    def forward(self, x): # (batch, in_channels, height, width)
        x = self.projection(x)  # (batch, n_embd, patches, patches)
        x = x.flatten(2)  # (batch, n_embd, n_patches) - "stacking" all patches
        x = x.transpose(1, 2)  # (batch, n_patches, n_embd)
        return x
    

class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
       
    def forward(self, q, k, v):
        B, T, E = k.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q  = self.q_attn(q)
        k  = self.k_attn(k)
        v  = self.v_attn(v)
        q = q.view(B, T, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, E // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, E) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

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


class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        out = self.ln_1(x)
        x = x + self.attn(out, out, out)
        
        out = self.ln_2(x)
        x = x + self.mlp(out)
        return x
    

class ViT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(config)
        num_patches = self.patch_embed.n_patches

        # [class] token. Learnable embedding whose state at the output
        # of the Transformer encoder serves as the image representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))

        # Position embeddings will be added to the patch embeddings
        # to retain positional information. Standard learnable 1D
        # position embeddings. One per token/patch + [class] token
        self.pos_embed = nn.Parameter(torch.zeros(
            1, 1 + num_patches, config.n_embd
        ))  # Could be implemented with nn.Embedding

        self.encoder = nn.ModuleDict(dict(
            transformer_blocks = nn.ModuleList([
                EncoderBlock(config) for _ in range(config.n_encoder_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.fc = nn.Linear(config.n_embd, config.num_classes, bias=False)
      
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.encoder.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_encoder_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):  # (batch, 1, 28, 28)
        batch_size = x.size()[0]

        # Transform the input with patch embedding
        emb = self.patch_embed(x)  # (batch, num_patches, n_embd)
        # Generate a [class] token for every sample in the batch
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, n_embd)
        # Concat the [class] token to the input
        emb = torch.cat((cls_token, emb), dim=1)  # (batch, 1 + num_patches, n_embd)
        # Apply the positional embedding
        emb = emb + self.pos_embed  # (batch, 1 + num_patches, n_embd)

        # THe input is ready (batch, 1 + num_patches, n_embd) => Encoder
        for block in self.encoder.transformer_blocks:
            emb = block(emb)
        enc_out = self.encoder.ln_f(emb)

        # Given the encoder output (batch, 1 + num_patches, n_embd)
        # Take only the [class] token for prediction
        cls_out = enc_out[:, 0]  # (batch, n_embd)
        y = self.fc(cls_out)  # (batch, num_classes)

        return y


def get_accuracy(model, dataloader, device):
    """
    Compute the accuracy of the model on the given CharactersDataset

    Args:
        model: the model to evaluate
        dataloader: the DataLoader object to use
        device: the device to run the model on

    Returns:
        The accuracy of the model on the given dataset
    """
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)

            logits = model(x)
            _, predicted_labels = torch.max(logits, 1)  # Get index of maximum logit as prediction
            correct += (predicted_labels == y).sum().item()
            total += len(x)
        
    model.train()
    return correct / total