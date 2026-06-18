import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

from .common import TransformerBlock, tokens_to_image


class PatchEmbedding(nn.Cell):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            pad_mode="valid",
            has_bias=True,
        )

    def construct(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = ops.reshape(x, (b, c, h * w))
        x = ops.transpose(x, (0, 2, 1))
        return x, (h, w)


class SETRTiny(nn.Cell):
    def __init__(
        self,
        num_classes,
        image_size=256,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.0,
        decoder_dim=96,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        pos = np.zeros((1, self.patch_embed.num_patches, embed_dim), dtype=np.float32)
        self.pos_embed = Parameter(Tensor(pos, ms.float32), name="pos_embed")
        self.pos_drop = nn.Dropout(p=float(drop_rate))
        self.blocks = nn.CellList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, sr_ratio=1)
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm((embed_dim,))
        self.decoder = nn.SequentialCell(
            nn.Conv2d(embed_dim, decoder_dim, kernel_size=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1, has_bias=True),
        )

    def construct(self, x):
        tokens, hw = self.patch_embed(x)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        for block in self.blocks:
            tokens = block(tokens, hw)
        tokens = self.norm(tokens)
        feat = tokens_to_image(tokens, hw[0], hw[1])
        logits = self.decoder(feat)
        return ops.interpolate(
            logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
