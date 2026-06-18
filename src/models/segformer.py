from mindspore import nn, ops

from .common import TransformerBlock, image_to_tokens, tokens_to_image


class OverlapPatchEmbed(nn.Cell):
    def __init__(self, in_channels, embed_dim, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=kernel_size // 2,
            has_bias=True,
        )
        self.norm = nn.LayerNorm((embed_dim,))

    def construct(self, x):
        x = self.proj(x)
        tokens, hw = image_to_tokens(x)
        tokens = self.norm(tokens)
        return tokens, hw


class MixVisionTransformer(nn.Cell):
    def __init__(
        self,
        embed_dims,
        depths,
        num_heads,
        sr_ratios,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.patch_embeds = nn.CellList()
        self.blocks = nn.CellList()
        self.norms = nn.CellList()

        in_channels = [3] + list(embed_dims[:-1])
        kernel_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]

        for i in range(4):
            dim = int(embed_dims[i])
            self.patch_embeds.append(
                OverlapPatchEmbed(in_channels[i], dim, kernel_sizes[i], strides[i])
            )
            stage_blocks = nn.CellList(
                [
                    TransformerBlock(
                        dim,
                        int(num_heads[i]),
                        mlp_ratio,
                        drop_rate,
                        int(sr_ratios[i]),
                    )
                    for _ in range(int(depths[i]))
                ]
            )
            self.blocks.append(stage_blocks)
            self.norms.append(nn.LayerNorm((dim,)))

    def construct(self, x):
        features = ()
        for i in range(4):
            tokens, hw = self.patch_embeds[i](x)
            for block in self.blocks[i]:
                tokens = block(tokens, hw)
            tokens = self.norms[i](tokens)
            x = tokens_to_image(tokens, hw[0], hw[1])
            features = features + (x,)
        return features


class SegFormerDecoder(nn.Cell):
    def __init__(self, embed_dims, decoder_dim, num_classes, image_size):
        super().__init__()
        self.image_size = int(image_size)
        self.proj = nn.CellList([nn.Dense(int(dim), decoder_dim) for dim in embed_dims])
        self.concat = ops.Concat(axis=1)
        self.fuse = nn.SequentialCell(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1, has_bias=True),
        )

    def _project_feature(self, x, index):
        b, c, h, w = x.shape
        tokens, _ = image_to_tokens(x)
        tokens = self.proj[index](tokens)
        tokens = ops.transpose(tokens, (0, 2, 1))
        return ops.reshape(tokens, (b, tokens.shape[1], h, w))

    def construct(self, features):
        target_hw = (features[0].shape[2], features[0].shape[3])
        outs = ()
        for i in range(4):
            x = self._project_feature(features[i], i)
            if i > 0:
                x = ops.interpolate(
                    x,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            outs = outs + (x,)
        logits = self.fuse(self.concat(outs))
        return ops.interpolate(
            logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )


class SegFormerTiny(nn.Cell):
    def __init__(
        self,
        num_classes,
        image_size=256,
        embed_dims=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        num_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratio=4.0,
        drop_rate=0.0,
        decoder_dim=128,
    ):
        super().__init__()
        self.encoder = MixVisionTransformer(
            embed_dims, depths, num_heads, sr_ratios, mlp_ratio, drop_rate
        )
        self.decoder = SegFormerDecoder(embed_dims, decoder_dim, num_classes, image_size)

    def construct(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class SegFormerEdge(nn.Cell):
    def __init__(
        self,
        num_classes,
        image_size=256,
        embed_dims=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        num_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratio=4.0,
        drop_rate=0.0,
        decoder_dim=128,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.encoder = MixVisionTransformer(
            embed_dims, depths, num_heads, sr_ratios, mlp_ratio, drop_rate
        )
        self.decoder = SegFormerDecoder(embed_dims, decoder_dim, num_classes, image_size)
        self.edge_head = nn.SequentialCell(
            nn.Conv2d(int(embed_dims[0]), decoder_dim // 2, kernel_size=3, pad_mode="pad", padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(decoder_dim // 2, 2, kernel_size=1, has_bias=True),
        )

    def construct(self, x):
        features = self.encoder(x)
        seg_logits = self.decoder(features)
        edge_logits = self.edge_head(features[0])
        edge_logits = ops.interpolate(
            edge_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return seg_logits, edge_logits
