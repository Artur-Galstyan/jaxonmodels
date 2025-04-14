def get_swin_model_order(version: int) -> list[str]:
    if version == 2:
        order = [
            "[0].features.layers[0].layers[0].weight",
            "[0].features.layers[0].layers[0].bias",
            "[0].features.layers[0].layers[2].weight",
            "[0].features.layers[0].layers[2].bias",
            "[0].features.layers[1].layers[0].norm1.weight",
            "[0].features.layers[1].layers[0].norm1.bias",
            "[0].features.layers[1].layers[0].attn.logit_scale",
            "[1][<flat index 1>]",
            "[1][<flat index 0>]",
            "[0].features.layers[1].layers[0].attn.qkv.weight",
            "[0].features.layers[1].layers[0].attn.qkv.bias",
            "[0].features.layers[1].layers[0].attn.proj.weight",
            "[0].features.layers[1].layers[0].attn.proj.bias",
            "[0].features.layers[1].layers[0].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[1].layers[0].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[1].layers[0].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[1].layers[0].norm2.weight",
            "[0].features.layers[1].layers[0].norm2.bias",
            "[0].features.layers[1].layers[0].mlp.layers[0].weight",
            "[0].features.layers[1].layers[0].mlp.layers[0].bias",
            "[0].features.layers[1].layers[0].mlp.layers[3].weight",
            "[0].features.layers[1].layers[0].mlp.layers[3].bias",
            "[0].features.layers[1].layers[1].norm1.weight",
            "[0].features.layers[1].layers[1].norm1.bias",
            "[0].features.layers[1].layers[1].attn.logit_scale",
            "[1][<flat index 3>]",
            "[1][<flat index 2>]",
            "[0].features.layers[1].layers[1].attn.qkv.weight",
            "[0].features.layers[1].layers[1].attn.qkv.bias",
            "[0].features.layers[1].layers[1].attn.proj.weight",
            "[0].features.layers[1].layers[1].attn.proj.bias",
            "[0].features.layers[1].layers[1].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[1].layers[1].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[1].layers[1].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[1].layers[1].norm2.weight",
            "[0].features.layers[1].layers[1].norm2.bias",
            "[0].features.layers[1].layers[1].mlp.layers[0].weight",
            "[0].features.layers[1].layers[1].mlp.layers[0].bias",
            "[0].features.layers[1].layers[1].mlp.layers[3].weight",
            "[0].features.layers[1].layers[1].mlp.layers[3].bias",
            "[0].features.layers[2].reduction.weight",
            "[0].features.layers[2].norm.weight",
            "[0].features.layers[2].norm.bias",
            "[0].features.layers[3].layers[0].norm1.weight",
            "[0].features.layers[3].layers[0].norm1.bias",
            "[0].features.layers[3].layers[0].attn.logit_scale",
            "[1][<flat index 5>]",
            "[1][<flat index 4>]",
            "[0].features.layers[3].layers[0].attn.qkv.weight",
            "[0].features.layers[3].layers[0].attn.qkv.bias",
            "[0].features.layers[3].layers[0].attn.proj.weight",
            "[0].features.layers[3].layers[0].attn.proj.bias",
            "[0].features.layers[3].layers[0].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[3].layers[0].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[3].layers[0].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[3].layers[0].norm2.weight",
            "[0].features.layers[3].layers[0].norm2.bias",
            "[0].features.layers[3].layers[0].mlp.layers[0].weight",
            "[0].features.layers[3].layers[0].mlp.layers[0].bias",
            "[0].features.layers[3].layers[0].mlp.layers[3].weight",
            "[0].features.layers[3].layers[0].mlp.layers[3].bias",
            "[0].features.layers[3].layers[1].norm1.weight",
            "[0].features.layers[3].layers[1].norm1.bias",
            "[0].features.layers[3].layers[1].attn.logit_scale",
            "[1][<flat index 7>]",
            "[1][<flat index 6>]",
            "[0].features.layers[3].layers[1].attn.qkv.weight",
            "[0].features.layers[3].layers[1].attn.qkv.bias",
            "[0].features.layers[3].layers[1].attn.proj.weight",
            "[0].features.layers[3].layers[1].attn.proj.bias",
            "[0].features.layers[3].layers[1].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[3].layers[1].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[3].layers[1].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[3].layers[1].norm2.weight",
            "[0].features.layers[3].layers[1].norm2.bias",
            "[0].features.layers[3].layers[1].mlp.layers[0].weight",
            "[0].features.layers[3].layers[1].mlp.layers[0].bias",
            "[0].features.layers[3].layers[1].mlp.layers[3].weight",
            "[0].features.layers[3].layers[1].mlp.layers[3].bias",
            "[0].features.layers[4].reduction.weight",
            "[0].features.layers[4].norm.weight",
            "[0].features.layers[4].norm.bias",
            "[0].features.layers[5].layers[0].norm1.weight",
            "[0].features.layers[5].layers[0].norm1.bias",
            "[0].features.layers[5].layers[0].attn.logit_scale",
            "[1][<flat index 9>]",
            "[1][<flat index 8>]",
            "[0].features.layers[5].layers[0].attn.qkv.weight",
            "[0].features.layers[5].layers[0].attn.qkv.bias",
            "[0].features.layers[5].layers[0].attn.proj.weight",
            "[0].features.layers[5].layers[0].attn.proj.bias",
            "[0].features.layers[5].layers[0].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[5].layers[0].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[5].layers[0].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[5].layers[0].norm2.weight",
            "[0].features.layers[5].layers[0].norm2.bias",
            "[0].features.layers[5].layers[0].mlp.layers[0].weight",
            "[0].features.layers[5].layers[0].mlp.layers[0].bias",
            "[0].features.layers[5].layers[0].mlp.layers[3].weight",
            "[0].features.layers[5].layers[0].mlp.layers[3].bias",
            "[0].features.layers[5].layers[1].norm1.weight",
            "[0].features.layers[5].layers[1].norm1.bias",
            "[0].features.layers[5].layers[1].attn.logit_scale",
            "[1][<flat index 11>]",
            "[1][<flat index 10>]",
            "[0].features.layers[5].layers[1].attn.qkv.weight",
            "[0].features.layers[5].layers[1].attn.qkv.bias",
            "[0].features.layers[5].layers[1].attn.proj.weight",
            "[0].features.layers[5].layers[1].attn.proj.bias",
            "[0].features.layers[5].layers[1].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[5].layers[1].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[5].layers[1].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[5].layers[1].norm2.weight",
            "[0].features.layers[5].layers[1].norm2.bias",
            "[0].features.layers[5].layers[1].mlp.layers[0].weight",
            "[0].features.layers[5].layers[1].mlp.layers[0].bias",
            "[0].features.layers[5].layers[1].mlp.layers[3].weight",
            "[0].features.layers[5].layers[1].mlp.layers[3].bias",
            "[0].features.layers[5].layers[2].norm1.weight",
            "[0].features.layers[5].layers[2].norm1.bias",
            "[0].features.layers[5].layers[2].attn.logit_scale",
            "[1][<flat index 13>]",
            "[1][<flat index 12>]",
            "[0].features.layers[5].layers[2].attn.qkv.weight",
            "[0].features.layers[5].layers[2].attn.qkv.bias",
            "[0].features.layers[5].layers[2].attn.proj.weight",
            "[0].features.layers[5].layers[2].attn.proj.bias",
            "[0].features.layers[5].layers[2].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[5].layers[2].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[5].layers[2].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[5].layers[2].norm2.weight",
            "[0].features.layers[5].layers[2].norm2.bias",
            "[0].features.layers[5].layers[2].mlp.layers[0].weight",
            "[0].features.layers[5].layers[2].mlp.layers[0].bias",
            "[0].features.layers[5].layers[2].mlp.layers[3].weight",
            "[0].features.layers[5].layers[2].mlp.layers[3].bias",
            "[0].features.layers[5].layers[3].norm1.weight",
            "[0].features.layers[5].layers[3].norm1.bias",
            "[0].features.layers[5].layers[3].attn.logit_scale",
            "[1][<flat index 15>]",
            "[1][<flat index 14>]",
            "[0].features.layers[5].layers[3].attn.qkv.weight",
            "[0].features.layers[5].layers[3].attn.qkv.bias",
            "[0].features.layers[5].layers[3].attn.proj.weight",
            "[0].features.layers[5].layers[3].attn.proj.bias",
            "[0].features.layers[5].layers[3].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[5].layers[3].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[5].layers[3].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[5].layers[3].norm2.weight",
            "[0].features.layers[5].layers[3].norm2.bias",
            "[0].features.layers[5].layers[3].mlp.layers[0].weight",
            "[0].features.layers[5].layers[3].mlp.layers[0].bias",
            "[0].features.layers[5].layers[3].mlp.layers[3].weight",
            "[0].features.layers[5].layers[3].mlp.layers[3].bias",
            "[0].features.layers[5].layers[4].norm1.weight",
            "[0].features.layers[5].layers[4].norm1.bias",
            "[0].features.layers[5].layers[4].attn.logit_scale",
            "[1][<flat index 17>]",
            "[1][<flat index 16>]",
            "[0].features.layers[5].layers[4].attn.qkv.weight",
            "[0].features.layers[5].layers[4].attn.qkv.bias",
            "[0].features.layers[5].layers[4].attn.proj.weight",
            "[0].features.layers[5].layers[4].attn.proj.bias",
            "[0].features.layers[5].layers[4].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[5].layers[4].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[5].layers[4].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[5].layers[4].norm2.weight",
            "[0].features.layers[5].layers[4].norm2.bias",
            "[0].features.layers[5].layers[4].mlp.layers[0].weight",
            "[0].features.layers[5].layers[4].mlp.layers[0].bias",
            "[0].features.layers[5].layers[4].mlp.layers[3].weight",
            "[0].features.layers[5].layers[4].mlp.layers[3].bias",
            "[0].features.layers[5].layers[5].norm1.weight",
            "[0].features.layers[5].layers[5].norm1.bias",
            "[0].features.layers[5].layers[5].attn.logit_scale",
            "[1][<flat index 19>]",
            "[1][<flat index 18>]",
            "[0].features.layers[5].layers[5].attn.qkv.weight",
            "[0].features.layers[5].layers[5].attn.qkv.bias",
            "[0].features.layers[5].layers[5].attn.proj.weight",
            "[0].features.layers[5].layers[5].attn.proj.bias",
            "[0].features.layers[5].layers[5].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[5].layers[5].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[5].layers[5].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[5].layers[5].norm2.weight",
            "[0].features.layers[5].layers[5].norm2.bias",
            "[0].features.layers[5].layers[5].mlp.layers[0].weight",
            "[0].features.layers[5].layers[5].mlp.layers[0].bias",
            "[0].features.layers[5].layers[5].mlp.layers[3].weight",
            "[0].features.layers[5].layers[5].mlp.layers[3].bias",
            "[0].features.layers[6].reduction.weight",
            "[0].features.layers[6].norm.weight",
            "[0].features.layers[6].norm.bias",
            "[0].features.layers[7].layers[0].norm1.weight",
            "[0].features.layers[7].layers[0].norm1.bias",
            "[0].features.layers[7].layers[0].attn.logit_scale",
            "[1][<flat index 21>]",
            "[1][<flat index 20>]",
            "[0].features.layers[7].layers[0].attn.qkv.weight",
            "[0].features.layers[7].layers[0].attn.qkv.bias",
            "[0].features.layers[7].layers[0].attn.proj.weight",
            "[0].features.layers[7].layers[0].attn.proj.bias",
            "[0].features.layers[7].layers[0].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[7].layers[0].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[7].layers[0].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[7].layers[0].norm2.weight",
            "[0].features.layers[7].layers[0].norm2.bias",
            "[0].features.layers[7].layers[0].mlp.layers[0].weight",
            "[0].features.layers[7].layers[0].mlp.layers[0].bias",
            "[0].features.layers[7].layers[0].mlp.layers[3].weight",
            "[0].features.layers[7].layers[0].mlp.layers[3].bias",
            "[0].features.layers[7].layers[1].norm1.weight",
            "[0].features.layers[7].layers[1].norm1.bias",
            "[0].features.layers[7].layers[1].attn.logit_scale",
            "[1][<flat index 23>]",
            "[1][<flat index 22>]",
            "[0].features.layers[7].layers[1].attn.qkv.weight",
            "[0].features.layers[7].layers[1].attn.qkv.bias",
            "[0].features.layers[7].layers[1].attn.proj.weight",
            "[0].features.layers[7].layers[1].attn.proj.bias",
            "[0].features.layers[7].layers[1].attn.cpb_mlp_ln1.weight",
            "[0].features.layers[7].layers[1].attn.cpb_mlp_ln1.bias",
            "[0].features.layers[7].layers[1].attn.cpb_mlp_ln2.weight",
            "[0].features.layers[7].layers[1].norm2.weight",
            "[0].features.layers[7].layers[1].norm2.bias",
            "[0].features.layers[7].layers[1].mlp.layers[0].weight",
            "[0].features.layers[7].layers[1].mlp.layers[0].bias",
            "[0].features.layers[7].layers[1].mlp.layers[3].weight",
            "[0].features.layers[7].layers[1].mlp.layers[3].bias",
            "[0].norm.weight",
            "[0].norm.bias",
            "[0].head.weight",
            "[0].head.bias",
        ]
    else:
        order = [
            "[0].features.layers[0].layers[0].weight",
            "[0].features.layers[0].layers[0].bias",
            "[0].features.layers[0].layers[2].weight",
            "[0].features.layers[0].layers[2].bias",
            "[0].features.layers[1].layers[0].norm1.weight",
            "[0].features.layers[1].layers[0].norm1.bias",
            "[0].features.layers[1].layers[0].attn.relative_position_bias_table",
            "[1][<flat index 0>]",
            "[0].features.layers[1].layers[0].attn.qkv.weight",
            "[0].features.layers[1].layers[0].attn.qkv.bias",
            "[0].features.layers[1].layers[0].attn.proj.weight",
            "[0].features.layers[1].layers[0].attn.proj.bias",
            "[0].features.layers[1].layers[0].norm2.weight",
            "[0].features.layers[1].layers[0].norm2.bias",
            "[0].features.layers[1].layers[0].mlp.layers[0].weight",
            "[0].features.layers[1].layers[0].mlp.layers[0].bias",
            "[0].features.layers[1].layers[0].mlp.layers[3].weight",
            "[0].features.layers[1].layers[0].mlp.layers[3].bias",
            # Stage 1, Block 1 (features.1.1 -> layers[1].layers[1])
            "[0].features.layers[1].layers[1].norm1.weight",
            "[0].features.layers[1].layers[1].norm1.bias",
            "[0].features.layers[1].layers[1].attn.relative_position_bias_table",
            "[1][<flat index 1>]",
            "[0].features.layers[1].layers[1].attn.qkv.weight",
            "[0].features.layers[1].layers[1].attn.qkv.bias",
            "[0].features.layers[1].layers[1].attn.proj.weight",
            "[0].features.layers[1].layers[1].attn.proj.bias",
            "[0].features.layers[1].layers[1].norm2.weight",
            "[0].features.layers[1].layers[1].norm2.bias",
            "[0].features.layers[1].layers[1].mlp.layers[0].weight",
            "[0].features.layers[1].layers[1].mlp.layers[0].bias",
            "[0].features.layers[1].layers[1].mlp.layers[3].weight",
            "[0].features.layers[1].layers[1].mlp.layers[3].bias",
            "[0].features.layers[2].reduction.weight",
            "[0].features.layers[2].norm.weight",
            "[0].features.layers[2].norm.bias",
            "[0].features.layers[3].layers[0].norm1.weight",
            "[0].features.layers[3].layers[0].norm1.bias",
            "[0].features.layers[3].layers[0].attn.relative_position_bias_table",
            "[1][<flat index 2>]",
            "[0].features.layers[3].layers[0].attn.qkv.weight",
            "[0].features.layers[3].layers[0].attn.qkv.bias",
            "[0].features.layers[3].layers[0].attn.proj.weight",
            "[0].features.layers[3].layers[0].attn.proj.bias",
            "[0].features.layers[3].layers[0].norm2.weight",
            "[0].features.layers[3].layers[0].norm2.bias",
            "[0].features.layers[3].layers[0].mlp.layers[0].weight",
            "[0].features.layers[3].layers[0].mlp.layers[0].bias",
            "[0].features.layers[3].layers[0].mlp.layers[3].weight",
            "[0].features.layers[3].layers[0].mlp.layers[3].bias",
            "[0].features.layers[3].layers[1].norm1.weight",
            "[0].features.layers[3].layers[1].norm1.bias",
            "[0].features.layers[3].layers[1].attn.relative_position_bias_table",
            "[1][<flat index 3>]",
            "[0].features.layers[3].layers[1].attn.qkv.weight",
            "[0].features.layers[3].layers[1].attn.qkv.bias",
            "[0].features.layers[3].layers[1].attn.proj.weight",
            "[0].features.layers[3].layers[1].attn.proj.bias",
            "[0].features.layers[3].layers[1].norm2.weight",
            "[0].features.layers[3].layers[1].norm2.bias",
            "[0].features.layers[3].layers[1].mlp.layers[0].weight",
            "[0].features.layers[3].layers[1].mlp.layers[0].bias",
            "[0].features.layers[3].layers[1].mlp.layers[3].weight",
            "[0].features.layers[3].layers[1].mlp.layers[3].bias",
            "[0].features.layers[4].reduction.weight",
            "[0].features.layers[4].norm.weight",
            "[0].features.layers[4].norm.bias",
            "[0].features.layers[5].layers[0].norm1.weight",
            "[0].features.layers[5].layers[0].norm1.bias",
            "[0].features.layers[5].layers[0].attn.relative_position_bias_table",
            "[1][<flat index 4>]",
            "[0].features.layers[5].layers[0].attn.qkv.weight",
            "[0].features.layers[5].layers[0].attn.qkv.bias",
            "[0].features.layers[5].layers[0].attn.proj.weight",
            "[0].features.layers[5].layers[0].attn.proj.bias",
            "[0].features.layers[5].layers[0].norm2.weight",
            "[0].features.layers[5].layers[0].norm2.bias",
            "[0].features.layers[5].layers[0].mlp.layers[0].weight",
            "[0].features.layers[5].layers[0].mlp.layers[0].bias",
            "[0].features.layers[5].layers[0].mlp.layers[3].weight",
            "[0].features.layers[5].layers[0].mlp.layers[3].bias",
            # Stage 3, Block 1 (features.5.1 -> layers[5].layers[1])
            "[0].features.layers[5].layers[1].norm1.weight",
            "[0].features.layers[5].layers[1].norm1.bias",
            "[0].features.layers[5].layers[1].attn.relative_position_bias_table",
            "[1][<flat index 5>]",
            "[0].features.layers[5].layers[1].attn.qkv.weight",
            "[0].features.layers[5].layers[1].attn.qkv.bias",
            "[0].features.layers[5].layers[1].attn.proj.weight",
            "[0].features.layers[5].layers[1].attn.proj.bias",
            "[0].features.layers[5].layers[1].norm2.weight",
            "[0].features.layers[5].layers[1].norm2.bias",
            "[0].features.layers[5].layers[1].mlp.layers[0].weight",
            "[0].features.layers[5].layers[1].mlp.layers[0].bias",
            "[0].features.layers[5].layers[1].mlp.layers[3].weight",
            "[0].features.layers[5].layers[1].mlp.layers[3].bias",
            # Stage 3, Block 2 (features.5.2 -> layers[5].layers[2])
            "[0].features.layers[5].layers[2].norm1.weight",
            "[0].features.layers[5].layers[2].norm1.bias",
            "[0].features.layers[5].layers[2].attn.relative_position_bias_table",
            "[1][<flat index 6>]",
            "[0].features.layers[5].layers[2].attn.qkv.weight",
            "[0].features.layers[5].layers[2].attn.qkv.bias",
            "[0].features.layers[5].layers[2].attn.proj.weight",
            "[0].features.layers[5].layers[2].attn.proj.bias",
            "[0].features.layers[5].layers[2].norm2.weight",
            "[0].features.layers[5].layers[2].norm2.bias",
            "[0].features.layers[5].layers[2].mlp.layers[0].weight",
            "[0].features.layers[5].layers[2].mlp.layers[0].bias",
            "[0].features.layers[5].layers[2].mlp.layers[3].weight",
            "[0].features.layers[5].layers[2].mlp.layers[3].bias",
            # Stage 3, Block 3 (features.5.3 -> layers[5].layers[3])
            "[0].features.layers[5].layers[3].norm1.weight",
            "[0].features.layers[5].layers[3].norm1.bias",
            "[0].features.layers[5].layers[3].attn.relative_position_bias_table",
            "[1][<flat index 7>]",
            "[0].features.layers[5].layers[3].attn.qkv.weight",
            "[0].features.layers[5].layers[3].attn.qkv.bias",
            "[0].features.layers[5].layers[3].attn.proj.weight",
            "[0].features.layers[5].layers[3].attn.proj.bias",
            "[0].features.layers[5].layers[3].norm2.weight",
            "[0].features.layers[5].layers[3].norm2.bias",
            "[0].features.layers[5].layers[3].mlp.layers[0].weight",
            "[0].features.layers[5].layers[3].mlp.layers[0].bias",
            "[0].features.layers[5].layers[3].mlp.layers[3].weight",
            "[0].features.layers[5].layers[3].mlp.layers[3].bias",
            # Stage 3, Block 4 (features.5.4 -> layers[5].layers[4])
            "[0].features.layers[5].layers[4].norm1.weight",
            "[0].features.layers[5].layers[4].norm1.bias",
            "[0].features.layers[5].layers[4].attn.relative_position_bias_table",
            "[1][<flat index 8>]",
            "[0].features.layers[5].layers[4].attn.qkv.weight",
            "[0].features.layers[5].layers[4].attn.qkv.bias",
            "[0].features.layers[5].layers[4].attn.proj.weight",
            "[0].features.layers[5].layers[4].attn.proj.bias",
            "[0].features.layers[5].layers[4].norm2.weight",
            "[0].features.layers[5].layers[4].norm2.bias",
            "[0].features.layers[5].layers[4].mlp.layers[0].weight",
            "[0].features.layers[5].layers[4].mlp.layers[0].bias",
            "[0].features.layers[5].layers[4].mlp.layers[3].weight",
            "[0].features.layers[5].layers[4].mlp.layers[3].bias",
            # Stage 3, Block 5 (features.5.5 -> layers[5].layers[5])
            "[0].features.layers[5].layers[5].norm1.weight",
            "[0].features.layers[5].layers[5].norm1.bias",
            "[0].features.layers[5].layers[5].attn.relative_position_bias_table",
            "[1][<flat index 9>]",
            "[0].features.layers[5].layers[5].attn.qkv.weight",
            "[0].features.layers[5].layers[5].attn.qkv.bias",
            "[0].features.layers[5].layers[5].attn.proj.weight",
            "[0].features.layers[5].layers[5].attn.proj.bias",
            "[0].features.layers[5].layers[5].norm2.weight",
            "[0].features.layers[5].layers[5].norm2.bias",
            "[0].features.layers[5].layers[5].mlp.layers[0].weight",
            "[0].features.layers[5].layers[5].mlp.layers[0].bias",
            "[0].features.layers[5].layers[5].mlp.layers[3].weight",
            "[0].features.layers[5].layers[5].mlp.layers[3].bias",
            # Downsample 3 (features.6 -> layers[6])
            "[0].features.layers[6].reduction.weight",
            "[0].features.layers[6].norm.weight",
            "[0].features.layers[6].norm.bias",
            # Stage 4, Block 0 (features.7.0 -> layers[7].layers[0])
            "[0].features.layers[7].layers[0].norm1.weight",
            "[0].features.layers[7].layers[0].norm1.bias",
            "[0].features.layers[7].layers[0].attn.relative_position_bias_table",
            "[1][<flat index 10>]",
            "[0].features.layers[7].layers[0].attn.qkv.weight",
            "[0].features.layers[7].layers[0].attn.qkv.bias",
            "[0].features.layers[7].layers[0].attn.proj.weight",
            "[0].features.layers[7].layers[0].attn.proj.bias",
            "[0].features.layers[7].layers[0].norm2.weight",
            "[0].features.layers[7].layers[0].norm2.bias",
            "[0].features.layers[7].layers[0].mlp.layers[0].weight",
            "[0].features.layers[7].layers[0].mlp.layers[0].bias",
            "[0].features.layers[7].layers[0].mlp.layers[3].weight",
            "[0].features.layers[7].layers[0].mlp.layers[3].bias",
            "[0].features.layers[7].layers[1].norm1.weight",
            "[0].features.layers[7].layers[1].norm1.bias",
            "[0].features.layers[7].layers[1].attn.relative_position_bias_table",
            "[1][<flat index 11>]",
            "[0].features.layers[7].layers[1].attn.qkv.weight",
            "[0].features.layers[7].layers[1].attn.qkv.bias",
            "[0].features.layers[7].layers[1].attn.proj.weight",
            "[0].features.layers[7].layers[1].attn.proj.bias",
            "[0].features.layers[7].layers[1].norm2.weight",
            "[0].features.layers[7].layers[1].norm2.bias",
            "[0].features.layers[7].layers[1].mlp.layers[0].weight",
            "[0].features.layers[7].layers[1].mlp.layers[0].bias",
            "[0].features.layers[7].layers[1].mlp.layers[3].weight",
            "[0].features.layers[7].layers[1].mlp.layers[3].bias",
            "[0].norm.weight",
            "[0].norm.bias",
            "[0].head.weight",
            "[0].head.bias",
        ]
    return order
