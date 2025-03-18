import gzip
import html
import os
from functools import lru_cache

import ftfy
import jax
import jax.numpy as jnp
import regex as re
from jaxtyping import Array, Float, Int, PRNGKeyArray

__all__ = [
    "multi_head_attention_forward",
    "clip_tokenize",
    "build_attention_mask",
    "canonical_mask",
]


def multi_head_attention_forward(
    query: Float[Array, "tgt_len d_model"],
    key: Float[Array, "src_len d_model"],
    value: Float[Array, "src_len d_model"],
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Float[Array, "3*d_model d_model"] | None = None,
    in_proj_bias: Float[Array, "3*d_model"] | None = None,
    bias_k: Float[Array, "1 d_model"] | None = None,
    bias_v: Float[Array, "1 d_model"] | None = None,
    add_zero_attn: bool = False,
    dropout_p: float = 0.0,
    out_proj_weight: Float[Array, "d_model d_model"] | None = None,
    out_proj_bias: Float[Array, "d_model"] | None = None,
    inference: bool = False,
    key_padding_mask: Float[Array, "src_len"] | None = None,
    attn_mask: Float[Array, "tgt_len src_len"] | None = None,
    need_weights: bool = True,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Float[Array, "d_model d_model"] | None = None,
    k_proj_weight: Float[Array, "d_model d_model"] | None = None,
    v_proj_weight: Float[Array, "d_model d_model"] | None = None,
    static_k: Float[Array, "src_len d_model"] | None = None,
    static_v: Float[Array, "src_len d_model"] | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    dropout_key: PRNGKeyArray | None = None,
) -> tuple[
    Float[Array, "tgt_len d_model"],
    Float[Array, "num_heads tgt_len src_len"] | Float[Array, "tgt_len src_len"] | None,
]:
    tgt_len, d_model = query.shape
    src_len, k_dim = key.shape
    value_len, v_dim = value.shape

    assert d_model == k_dim == v_dim == embed_dim_to_check, (
        "Embedding dimensions must match"
    )

    assert src_len == value_len, "Key and value must have the same sequence length"

    head_dim = d_model // num_heads
    assert head_dim * num_heads == d_model, "embed_dim must be divisible by num_heads"

    if dropout_p > 0.0:
        assert dropout_key is not None, (
            "dropout_key must be provided if dropout_p > 0.0"
        )

    if use_separate_proj_weight:
        # When using separate projection weights for q, k, v
        assert q_proj_weight is not None, (
            "q_proj_weight should not be None when use_separate_proj_weight=True"
        )
        assert k_proj_weight is not None, (
            "k_proj_weight should not be None when use_separate_proj_weight=True"
        )
        assert v_proj_weight is not None, (
            "v_proj_weight should not be None when use_separate_proj_weight=True"
        )

        q = query @ q_proj_weight.T

        if static_k is None:
            k = key @ k_proj_weight.T
        else:
            k = static_k
            src_len, _ = k.shape

        if static_v is None:
            v = value @ v_proj_weight.T
        else:
            v = static_v
            value_len, _ = v.shape

        if in_proj_bias is not None:
            q_bias, k_bias, v_bias = jnp.split(in_proj_bias, 3)
            q = q + q_bias
            k = k + k_bias
            v = v + v_bias

    else:
        assert in_proj_weight is not None, (
            "in_proj_weight should not be None when use_separate_proj_weight=False"
        )

        q_proj_weight_part, k_proj_weight_part, v_proj_weight_part = jnp.split(
            in_proj_weight, 3
        )

        q = query @ q_proj_weight_part.T

        if static_k is None:
            k = key @ k_proj_weight_part.T
        else:
            k = static_k
            src_len, _ = static_k.shape

        if static_v is None:
            v = value @ v_proj_weight_part.T
        else:
            v = static_v
            value_len, _ = static_v.shape

        if in_proj_bias is not None:
            q_bias, k_bias, v_bias = jnp.split(in_proj_bias, 3)
            q = q + q_bias
            k = k + k_bias
            v = v + v_bias

    assert src_len == value_len

    q = q.reshape(tgt_len, num_heads, head_dim)
    k = k.reshape(src_len, num_heads, head_dim)
    v = v.reshape(src_len, num_heads, head_dim)

    if add_zero_attn:
        zero_attn_shape = (1, num_heads, head_dim)
        k_zeros = jnp.zeros(zero_attn_shape)
        v_zeros = jnp.zeros(zero_attn_shape)

        k = jnp.concatenate([k, k_zeros], axis=0)
        v = jnp.concatenate([v, v_zeros], axis=0)

        src_len += 1
        value_len += 1

    if bias_k is not None and bias_v is not None:
        bias_k = bias_k.reshape(1, num_heads, head_dim)
        bias_v = bias_v.reshape(1, num_heads, head_dim)

        k = jnp.concatenate([k, bias_k], axis=0)
        v = jnp.concatenate([v, bias_v], axis=0)

        src_len += 1
        value_len += 1

    assert src_len == value_len

    # [tgt_len, num_heads, head_dim] → [num_heads, tgt_len, head_dim]
    q = jnp.transpose(q, (1, 0, 2))

    # [src_len, num_heads, head_dim] → [num_heads, src_len, head_dim]
    k = jnp.transpose(k, (1, 0, 2))
    v = jnp.transpose(v, (1, 0, 2))

    scale = jnp.sqrt(head_dim)
    attn_output_weights = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / scale

    if key_padding_mask is not None:
        padding_mask = key_padding_mask.reshape(1, 1, src_len)
        padding_mask = jnp.repeat(padding_mask, num_heads, axis=0)
        padding_mask = jnp.repeat(padding_mask, tgt_len, axis=1)
        attn_output_weights = jnp.where(
            padding_mask, float("-inf"), attn_output_weights
        )

    if attn_mask is not None:
        # [tgt_len, src_len] -> [num_heads, tgt_len, src_len]
        mask = attn_mask.reshape(1, tgt_len, src_len)
        mask = jnp.repeat(mask, num_heads, axis=0)
        attn_output_weights = attn_output_weights + mask

    if is_causal:
        causal_mask = jnp.triu(jnp.ones((tgt_len, src_len)), k=1)
        causal_mask = (causal_mask == 1).reshape(1, tgt_len, src_len)
        causal_mask = jnp.repeat(causal_mask, num_heads, axis=0)
        attn_output_weights = jnp.where(causal_mask, float("-inf"), attn_output_weights)

    # [num_heads, tgt_len, src_len]
    attn_output_weights = jax.nn.softmax(attn_output_weights, axis=-1)

    if dropout_p > 0.0 and not inference:
        assert dropout_key is not None, (
            "dropout_key required because dropout_p > 0.0 and training"
        )
        dropout_mask = jax.random.bernoulli(
            dropout_key, 1 - dropout_p, attn_output_weights.shape
        )
        scale = 1.0 / (1.0 - dropout_p)
        attn_output_weights = attn_output_weights * dropout_mask * scale

    attn_output = jnp.matmul(attn_output_weights, v)
    attn_output = jnp.transpose(attn_output, (1, 0, 2))
    attn_output = attn_output.reshape(tgt_len, d_model)

    assert out_proj_weight is not None, "out_proj_weight must be provided"
    attn_output = attn_output @ out_proj_weight.T

    if out_proj_bias is not None:
        attn_output = attn_output + out_proj_bias

    if need_weights:
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(axis=0)
        return attn_output, attn_output_weights
    else:
        return attn_output, None


@lru_cache()
def default_bpe():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz"
    )


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you
    want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K
    for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:  # noqa
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text


def clip_tokenize(
    texts: str | list[str], context_length: int = 77, truncate: bool = False
) -> Int[Array, "n c"]:
    _tokenizer = SimpleTokenizer()
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = jnp.zeros(shape=(len(all_tokens), context_length), dtype=jnp.int32)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result = result.at[i, : len(tokens)].set(jnp.array(tokens))

    return result


def build_attention_mask(context_length: int):
    mask = jnp.tril(jnp.zeros((context_length, context_length)))
    upper = jnp.triu(jnp.full((context_length, context_length), float("-inf")), k=1)

    mask = mask + upper
    return mask


def canonical_mask(
    mask,
    mask_name,
    other_name="",
    other_type=None,
    target_type=jnp.float32,
    other_mask=None,
    check_other=True,
):
    if mask is None:
        return None
    if mask.dtype == bool:
        additive_mask = jnp.where(mask, -jnp.inf, 0.0).astype(target_type)
        return additive_mask
    elif jnp.issubdtype(mask.dtype, jnp.integer) or jnp.issubdtype(
        mask.dtype, jnp.floating
    ):
        return mask.astype(target_type)
    else:
        raise TypeError(
            f"{mask_name} must be bool, int, or float tensor, but got {mask.dtype}"
        )


def canonical_key_padding_mask(
    key_padding_mask, attn_mask=None, query_dtype=jnp.float32
):
    """Wrapper for canonicalizing key_padding_mask"""
    return canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_name="attn_mask",
        other_mask=attn_mask,
        target_type=query_dtype,
    )


def canonical_attn_mask(attn_mask, query_dtype=jnp.float32):
    """Wrapper for canonicalizing attn_mask"""
    return canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query_dtype,
        check_other=False,
    )
