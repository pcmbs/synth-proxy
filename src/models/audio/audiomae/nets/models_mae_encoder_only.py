# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# Import modified from https://github.com/facebookresearch/AudioMAE/tree/main

from functools import partial

import torch
from torch import nn

from . import (
    Block,
    # SwinTransformerBlock,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_flexible,
    PatchEmbed_new,
    PatchEmbed_org,
)


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=10,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        # decoder_depth=8,
        # decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        # norm_pix_loss=False,
        audio_exp=False,
        # alpha=0.0,
        # temperature=0.2,
        # mode=0,
        contextual_depth=8,
        use_custom_patch=False,
        # split_pos=False,
        pos_trainable=False,
        # use_nce=False,
        # beta=4.0,
        # decoder_mode=0,
        # mask_t_prob=0.6,
        # mask_f_prob=0.5,
        # mask_2d=False,
        # epoch=0,
        # no_shift=False,
    ):
        super().__init__()

        self.audio_exp = audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if use_custom_patch:
            print(f"Use custom patch_emb with patch size: {patch_size}, stride: {stride}")
            self.patch_embed = PatchEmbed_new(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                stride=stride,
            )
        else:
            self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable
        )  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(
                self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True
            )
        else:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.patch_embed.num_patches**0.5),
                cls_token=True,
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder_no_mask(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks

        # pcmbs: no contextual embeddings, i.e., only last layer's output followed
        # by a layer norm (default behavior?)
        if self.contextual_depth == -1:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x

        # contextual embeddings: mean over the last
        # "encoder_depth - (contextual_depth)" layers (without layer norm)
        # rewrote the following to be more memory efficient
        else:
            # contextual_embs_old = []
            contextual_embs = torch.zeros_like(x).to(x.device)
            for n, blk in enumerate(self.blocks):
                x = blk(x)
                if n >= self.contextual_depth:
                    # contextual_embs_old.append(self.norm(x))
                    contextual_embs += self.norm(x)
            # contextual_emb_old = torch.stack(contextual_embs_old, dim=0).mean(dim=0)
            contextual_embs = contextual_embs / (self.encoder_depth - self.contextual_depth)

            return contextual_embs


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        # decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        # decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        # decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        # decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
