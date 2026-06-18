import mindspore as ms
from mindspore import nn, ops


class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim, drop_rate=0.0):
        super().__init__()
        self.fc1 = nn.Dense(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=float(drop_rate))
        self.fc2 = nn.Dense(hidden_dim, dim)
        self.drop2 = nn.Dropout(p=float(drop_rate))

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MultiHeadSelfAttention(nn.Cell):
    def __init__(self, dim, num_heads, drop_rate=0.0, sr_ratio=1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = int(sr_ratio)

        self.q = nn.Dense(dim, dim)
        self.k = nn.Dense(dim, dim)
        self.v = nn.Dense(dim, dim)
        self.proj = nn.Dense(dim, dim)
        self.softmax = nn.Softmax(axis=-1)
        self.attn_drop = nn.Dropout(p=float(drop_rate))
        self.proj_drop = nn.Dropout(p=float(drop_rate))

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim,
                dim,
                kernel_size=self.sr_ratio,
                stride=self.sr_ratio,
                pad_mode="valid",
                has_bias=True,
            )
            self.norm_sr = nn.LayerNorm((dim,))
        else:
            self.sr = None
            self.norm_sr = None

    def construct(self, x, hw=None):
        b, n, c = x.shape
        q = self.q(x)
        q = ops.reshape(q, (b, n, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        kv_source = x
        if self.sr_ratio > 1 and hw is not None:
            h, w = hw
            x_image = ops.transpose(x, (0, 2, 1))
            x_image = ops.reshape(x_image, (b, c, h, w))
            x_image = self.sr(x_image)
            rh, rw = x_image.shape[2], x_image.shape[3]
            kv_source = ops.reshape(x_image, (b, c, rh * rw))
            kv_source = ops.transpose(kv_source, (0, 2, 1))
            kv_source = self.norm_sr(kv_source)

        k = self.k(kv_source)
        v = self.v(kv_source)
        k_len = k.shape[1]
        k = ops.reshape(k, (b, k_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (b, k_len, self.num_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n, c))
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Cell):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_rate=0.0, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm((dim,))
        self.attn = MultiHeadSelfAttention(dim, num_heads, drop_rate, sr_ratio)
        self.norm2 = nn.LayerNorm((dim,))
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, hidden_dim, drop_rate)

    def construct(self, x, hw=None):
        x = x + self.attn(self.norm1(x), hw)
        x = x + self.mlp(self.norm2(x))
        return x


def tokens_to_image(tokens, h, w):
    b, _, c = tokens.shape
    x = ops.transpose(tokens, (0, 2, 1))
    return ops.reshape(x, (b, c, h, w))


def image_to_tokens(x):
    b, c, h, w = x.shape
    x = ops.reshape(x, (b, c, h * w))
    x = ops.transpose(x, (0, 2, 1))
    return x, (h, w)
