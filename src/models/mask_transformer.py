import math

import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

from .common import FeedForward, image_to_tokens
from .segformer import MixVisionTransformer


class CrossAttention(nn.Cell):
    def __init__(self, dim, num_heads, drop_rate=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.scale = self.head_dim ** -0.5

        self.q = nn.Dense(dim, dim)
        self.k = nn.Dense(dim, dim)
        self.v = nn.Dense(dim, dim)
        self.proj = nn.Dense(dim, dim)
        self.softmax = nn.Softmax(axis=-1)
        self.attn_drop = nn.Dropout(p=float(drop_rate))
        self.proj_drop = nn.Dropout(p=float(drop_rate))

    def construct(self, query, context):
        b, q_len, _ = query.shape
        c_len = context.shape[1]

        q = self.q(query)
        k = self.k(context)
        v = self.v(context)

        q = ops.reshape(q, (b, q_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (b, c_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (b, c_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, q_len, self.dim))
        out = self.proj(out)
        return self.proj_drop(out)


class QueryDecoderLayer(nn.Cell):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_rate=0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm((dim,))
        self.self_attn = CrossAttention(dim, num_heads, drop_rate)
        self.norm_cross = nn.LayerNorm((dim,))
        self.norm_context = nn.LayerNorm((dim,))
        self.cross_attn = CrossAttention(dim, num_heads, drop_rate)
        self.norm_mlp = nn.LayerNorm((dim,))
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), drop_rate)

    def construct(self, query, context):
        query = query + self.self_attn(self.norm_self(query), self.norm_self(query))
        query = query + self.cross_attn(self.norm_cross(query), self.norm_context(context))
        query = query + self.mlp(self.norm_mlp(query))
        return query


class SegFormerPixelDecoder(nn.Cell):
    def __init__(self, embed_dims, decoder_dim, mask_dim):
        super().__init__()
        self.proj = nn.CellList([nn.Dense(int(dim), decoder_dim) for dim in embed_dims])
        self.concat = ops.Concat(axis=1)
        self.fuse = nn.SequentialCell(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(decoder_dim, mask_dim, kernel_size=1, has_bias=True),
        )

    def _project_feature(self, x, index):
        b, _, h, w = x.shape
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
        return self.fuse(self.concat(outs))


class MaskQuerySegmentationHead(nn.Cell):
    def __init__(
        self,
        num_classes,
        image_size,
        hidden_dim=128,
        mask_dim=128,
        num_queries=100,
        num_heads=4,
        decoder_layers=3,
        mlp_ratio=4.0,
        drop_rate=0.0,
        use_task_token=False,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)
        self.use_task_token = bool(use_task_token)

        query = np.random.normal(0.0, 0.02, (1, self.num_queries, hidden_dim)).astype(np.float32)
        self.query_embed = Parameter(Tensor(query, ms.float32), name="query_embed")
        if self.use_task_token:
            task = np.random.normal(0.0, 0.02, (1, 1, hidden_dim)).astype(np.float32)
            self.task_token = Parameter(Tensor(task, ms.float32), name="semantic_task_token")
            self.task_proj = nn.Dense(hidden_dim, hidden_dim)
        else:
            self.task_token = None
            self.task_proj = None

        self.context_proj = nn.Conv2d(mask_dim, hidden_dim, kernel_size=1, has_bias=True)
        self.layers = nn.CellList(
            [
                QueryDecoderLayer(hidden_dim, num_heads, mlp_ratio, drop_rate)
                for _ in range(int(decoder_layers))
            ]
        )
        self.norm = nn.LayerNorm((hidden_dim,))
        self.class_head = nn.Dense(hidden_dim, int(num_classes))
        self.mask_embed = nn.SequentialCell(
            nn.Dense(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, mask_dim),
        )

    def _queries(self, batch_size):
        query = ops.tile(self.query_embed, (batch_size, 1, 1))
        if self.use_task_token:
            task = ops.tile(self.task_token, (batch_size, self.num_queries, 1))
            query = query + self.task_proj(task)
        return query

    def construct(self, mask_features):
        b, mask_dim, h, w = mask_features.shape
        context = self.context_proj(mask_features)
        context, _ = image_to_tokens(context)

        query = self._queries(b)
        for layer in self.layers:
            query = layer(query, context)
        query = self.norm(query)

        class_logits = self.class_head(query)
        mask_embed = self.mask_embed(query)
        pixel_tokens = ops.reshape(mask_features, (b, mask_dim, h * w))
        mask_logits = ops.matmul(mask_embed, pixel_tokens)
        mask_logits = ops.reshape(mask_logits, (b, self.num_queries, h, w))

        sem_logits = ops.matmul(
            ops.transpose(class_logits, (0, 2, 1)),
            ops.reshape(mask_logits, (b, self.num_queries, h * w)),
        )
        sem_logits = sem_logits / math.sqrt(float(self.num_queries))
        sem_logits = ops.reshape(sem_logits, (b, class_logits.shape[2], h, w))
        return ops.interpolate(
            sem_logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )


class Mask2FormerTiny(nn.Cell):
    def __init__(
        self,
        num_classes,
        image_size=512,
        embed_dims=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        num_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratio=4.0,
        drop_rate=0.0,
        decoder_dim=128,
        mask_dim=128,
        query_dim=128,
        num_queries=100,
        query_heads=4,
        query_layers=3,
    ):
        super().__init__()
        self.encoder = MixVisionTransformer(
            embed_dims, depths, num_heads, sr_ratios, mlp_ratio, drop_rate
        )
        self.pixel_decoder = SegFormerPixelDecoder(embed_dims, decoder_dim, mask_dim)
        self.query_head = MaskQuerySegmentationHead(
            num_classes=num_classes,
            image_size=image_size,
            hidden_dim=query_dim,
            mask_dim=mask_dim,
            num_queries=num_queries,
            num_heads=query_heads,
            decoder_layers=query_layers,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            use_task_token=False,
        )

    def construct(self, x):
        features = self.encoder(x)
        mask_features = self.pixel_decoder(features)
        return self.query_head(mask_features)


class OneFormerTiny(nn.Cell):
    def __init__(
        self,
        num_classes,
        image_size=512,
        embed_dims=(32, 64, 160, 256),
        depths=(2, 2, 2, 2),
        num_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratio=4.0,
        drop_rate=0.0,
        decoder_dim=128,
        mask_dim=128,
        query_dim=128,
        num_queries=100,
        query_heads=4,
        query_layers=3,
    ):
        super().__init__()
        self.encoder = MixVisionTransformer(
            embed_dims, depths, num_heads, sr_ratios, mlp_ratio, drop_rate
        )
        self.pixel_decoder = SegFormerPixelDecoder(embed_dims, decoder_dim, mask_dim)
        self.query_head = MaskQuerySegmentationHead(
            num_classes=num_classes,
            image_size=image_size,
            hidden_dim=query_dim,
            mask_dim=mask_dim,
            num_queries=num_queries,
            num_heads=query_heads,
            decoder_layers=query_layers,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            use_task_token=True,
        )

    def construct(self, x):
        features = self.encoder(x)
        mask_features = self.pixel_decoder(features)
        return self.query_head(mask_features)
