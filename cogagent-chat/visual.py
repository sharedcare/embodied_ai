from typing import Any
import mlx
import mlx.core as mx
import mlx.nn as nn
from argparse import Namespace
from transformers.activations import ACT2FN


def memory_efficient_attention(query: mx.array, key: mx.array, value: mx.array, attn_bias=None, p=0., scale=None) -> mx.array:
    """Refer to pytorch implementation of memory-efficient attention:
    https://facebookresearch.github.io/xformers/components/ops.html#module-xformers.ops

    """
    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5

    # (B, L, H, D) -> (B, H, L, D)
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
    attn = query @ key.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = mx.softmax(attn, axis=-1)
    attn = nn.Dropout(p)(attn)
    # (B, H, L, L) @ (B, H, L, D) -> (B, H, L, D)
    attn = attn @ value
    # (B, H, L, D) -> (B, L, H, D)
    out = attn.transpose(1, 2)
    return out


class PatchEmbedding(nn.Module):
    """
    """
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_embedding = mx.zeros([1, config.hidden_size])
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def __call__(self, images: mx.array) -> mx.array:
        # (B, C, H, W) -> (B, L, D)
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = mx.broadcast_to(self.cls_embedding, [x.shape[0], -1, -1])
        x = mx.concatenate((cls_token, x), dim=1)
        x += mx.expand_dims(self.position_embedding.weight, axis=0)
        return x


class Attention(nn.Module):
    """
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(p=config.dropout_prob)

    def __call__(self, x: mx.array) -> mx.array:
        # (B, L, D) -> (B, L, D)
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).transpose(
            2, 0, 1, 3, 4
        )  # 3, B, L, H, D
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = memory_efficient_attention(
            q, k, v, scale=self.scale,
        )
        output = self.dense(out.reshape(B, L, -1))
        output = self.output_dropout(output)
        return output

    def attention(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        attn_weights = q * self.scale @ k.transpose(-2, -1)
        attn_weights = mx.softmax(attn_weights, axis=-1)
        output = attn_weights @ v
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [TransformerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.gelu
        self.act2 = nn.silu
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=vision_config.hidden_size)
        self.boi = mx.zeros([1, 1, config.hidden_size])
        self.eoi = mx.zeros([1, 1, config.hidden_size])
        self.pos_embed = mx.zeros(
            [(vision_config.image_size // vision_config.patch_size) ** 2,
            vision_config.hidden_size],
        )

    def __call__(self, images: mx.array) -> mx.array:
        # (B, C, H, W) -> (B, L, D)
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]
        x = self.linear_proj(x + mx.expand_dims(self.pos_embed, axis=0))
        boi = mx.broadcast_to(self.boi, [x.shape[0], -1, -1])
        eoi = mx.broadcast_to(self.eoi, [x.shape[0], -1, -1])
        x = mx.concatenate((boi, x, eoi), axis=1)
        return x
