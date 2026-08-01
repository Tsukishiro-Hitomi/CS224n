"""
A bare-bones GPT-2 style transformer.
"""

import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int
from torch.nn.functional import softmax
from dataclasses import dataclass
from einops import rearrange
from transformers import GPT2LMHeadModel
import huggingface_hub

from utils import state_dict_converter


# TODO: Add in attention mask to the entire assignment
# TODO: Maybe add KV caching


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class CausalAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Using attention dim from attention is all you need
        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)

        #self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO, complete 
        # compute Q K V
        # shape: (N, L, h * d)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        N, L, H = x.shape
        d = self.d_attention
        h = H // d

        # (N, h, L, d)
        Q = Q.reshape(N, L, h, d)
        Q = Q.permute(0, 2, 1, 3)

        K = K.reshape(N, L, h, d)
        K = K.permute(0, 2, 1, 3)

        V = V.reshape(N, L, h, d)
        V = V.permute(0, 2, 1, 3)

        # scores & attn_weights: (N, h, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(d))
        scores = scores.masked_fill(self.causal_mask[:, :, :L, :L] == 0, float('-inf'))
        attn_weights = F.softmax(scores, -1)

        # context: (N, h, L, d)
        context = torch.matmul(attn_weights, V)

        # result: (N, L, h * d)
        # .contiguous: after 'permute' operation: the matrix maybe incontiguous in memory.
        result = context.permute(0, 2, 1, 3).contiguous()
        result = result.reshape(N, L, h * d)

        # O: (h * d, h * d)
        result = self.W_o(result)

        return result



class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO, complete
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
        

class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO complete
        x = x + self.attention(self.pre_layer_norm(x))
        x = x + self.mlp(self.post_layer_norm(x))
        return x


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:

        # TODO, complete
        # (B, L)->(B, L, D)
        B, L = x.shape
        token_emb = self.embeddings(x)
        positions = torch.arange(L, device=x.device)
        pos_emb = self.position_embeddings(positions)
        out = token_emb + pos_emb

        for block in self.backbone:
            out = block(out)
        out = self.final_layer_norm(out)
        # (B, L, D)->(B, L, V)
        out = self.lm_head(out)
        return out

    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        num_new_tokens: int,
    ) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:

        # TODO, complete
        B, L = x.shape
        for i in range(num_new_tokens):
            # (B, L, V)
            logits = self.forward(x)
            # (B, V) 
            # only take the last word's prediction
            next_logit = logits[:, -1, :]
            # (B, 1)
            new_token = torch.argmax(next_logit, dim=-1, keepdim=True)
            x = torch.cat([x, new_token], dim=1)
        return x


    def get_loss_on_batch(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"], 
    ) -> Float[Tensor, ""]:

        # TODO, complete
        B, L = input_ids.shape
        # (B, L-1)
        inputs = input_ids[:, :-1]
        # (B, L-1)
        targets = input_ids[:, 1:]

        # (B, L-1, V)
        logits = self.forward(inputs)
        _, _, V = logits.shape

        # (B * (L - 1), v)
        logits = logits.reshape(B * (L - 1), V)
        targets = targets.reshape(B * (L - 1), )

        return F.cross_entropy(logits, targets)


    @classmethod
    def from_pretrained(cls):
        """
        We simply always load up the GPT-2 model
        """

        # Config for GPT-2
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        # Load weights from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: Dict[str, Tensor] = state_dict_converter(model_hf.state_dict())

        model.load_state_dict(converted_state_dict)

        return model


if __name__ == "__main__":

    huggingface_hub.login()
    
    model = Transformer.from_pretrained()
