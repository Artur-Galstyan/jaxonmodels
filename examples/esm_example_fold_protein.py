import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

SEQUENCE_VOCAB = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "|",
    "<mask>",
]


def load_models():
    print("[1/7] Loading JAX ESM3 model...")
    from jaxonmodels.models.esm import ESM3 as JaxESM3

    jax_esm3 = JaxESM3.from_pretrained("esm3_open")

    print("[2/7] Loading PyTorch ESM3 model...")
    from esm.models.esm3 import ESM3 as TorchESM3  # ty:ignore[unresolved-import]

    torch_esm3 = TorchESM3.from_pretrained("esm3-open")

    print("[3/7] Loading structure decoders...")
    from esm.pretrained import ESM3_structure_decoder_v0  # ty:ignore[unresolved-import]

    from jaxonmodels.models.esm import StructureTokenDecoder as JaxDecoder

    jax_decoder = JaxDecoder.from_pretrained()
    torch_decoder = ESM3_structure_decoder_v0("cpu")

    return jax_esm3, torch_esm3, jax_decoder, torch_decoder


@eqx.filter_jit
def jit_forward(model, sequence_tokens):
    return model(sequence_tokens=sequence_tokens)


def encode_sequence(sequence, torch_esm3):
    from esm.sdk.api import ESMProtein  # ty:ignore[unresolved-import]

    protein = ESMProtein(sequence=sequence)
    protein_tensor = torch_esm3.encode(protein)
    return protein_tensor.sequence


def assert_forward_pass(jax_esm3, torch_esm3, sequence_tokens):
    print("\n[4/7] Asserting forward pass numerical equivalence...")

    with torch.no_grad():
        torch_out = torch_esm3.forward(sequence_tokens=sequence_tokens.unsqueeze(0))

    jax_seq_tokens = jnp.array(sequence_tokens.numpy())
    jax_out = jax_esm3(sequence_tokens=jax_seq_tokens)

    torch_seq_logits = torch_out.sequence_logits.squeeze(0).numpy()
    jax_seq_logits = np.array(jax_out[0])
    torch_struct_logits = torch_out.structure_logits.squeeze(0).numpy()
    jax_struct_logits = np.array(jax_out[1])

    seq_max_diff = np.abs(torch_seq_logits - jax_seq_logits).max()
    struct_max_diff = np.abs(torch_struct_logits - jax_struct_logits).max()

    print(f"       Sequence logits  max diff: {seq_max_diff:.6f}")
    print(f"       Structure logits max diff: {struct_max_diff:.6f}")

    assert np.allclose(jax_seq_logits, torch_seq_logits, atol=0.05), (
        f"Sequence logits diverged! max diff: {seq_max_diff}"
    )
    assert np.allclose(jax_struct_logits, torch_struct_logits, atol=0.05), (
        f"Structure logits diverged! max diff: {struct_max_diff}"
    )
    print("       ✓ Forward pass outputs match!")


def assert_decoder(jax_decoder, torch_decoder, seq_len):
    print("\n[5/7] Asserting structure decoder numerical equivalence...")

    np.random.seed(123)
    test_tokens = np.random.randint(0, 4096, size=(1, seq_len)).astype(np.int64)
    test_tokens[:, 0] = 4098
    test_tokens[:, -1] = 4097

    with torch.no_grad():
        torch_out = torch_decoder.decode(torch.from_numpy(test_tokens))
    jax_out = jax_decoder.decode(jnp.array(test_tokens[0]))

    bb_diff = np.abs(
        torch_out["bb_pred"].numpy()[0] - np.array(jax_out["bb_pred"])
    ).max()
    plddt_diff = np.abs(
        torch_out["plddt"].numpy()[0] - np.array(jax_out["plddt"])
    ).max()
    ptm_diff = np.abs(torch_out["ptm"].numpy().item() - np.array(jax_out["ptm"]).item())

    print(f"       Backbone coords max diff: {bb_diff:.6f}")
    print(f"       pLDDT max diff:           {plddt_diff:.6f}")
    print(f"       pTM diff:                 {ptm_diff:.6f}")

    assert bb_diff < 1.0, f"Backbone coords diverged! max diff: {bb_diff}"
    assert plddt_diff < 0.1, f"pLDDT diverged! max diff: {plddt_diff}"
    assert ptm_diff < 0.1, f"pTM diverged! diff: {ptm_diff}"
    print("       ✓ Structure decoder outputs match!")


def fold_and_validate(jax_esm3, jax_decoder, sequence):
    print("\n[6/7] Folding protein with JAX ESM3...")
    from jaxonmodels.models.esm import fold_protein

    key = jax.random.key(42)
    result = fold_protein(
        jax_esm3,
        jax_decoder,
        sequence,
        num_steps=8,
        temperature=0.7,
        key=key,
    )

    bb_coords = np.array(result["bb_coords"])
    plddt = np.array(result["plddt"])
    ptm = float(np.array(result["ptm"]))

    bos_eos_mask = np.ones(len(plddt), dtype=bool)
    bos_eos_mask[0] = False
    bos_eos_mask[-1] = False
    mean_plddt = plddt[bos_eos_mask].mean()

    print(f"       Backbone coords shape: {bb_coords.shape}")
    print(f"       Mean pLDDT:            {mean_plddt:.4f}")
    print(f"       pTM:                   {ptm:.4f}")

    ca_coords = bb_coords[1:-1, 1, :]
    dists = np.sqrt(((ca_coords[1:] - ca_coords[:-1]) ** 2).sum(axis=-1))
    mean_ca_dist = dists.mean()
    print(f"       Mean CA-CA distance:   {mean_ca_dist:.4f} Å")

    assert 3.0 < mean_ca_dist < 4.5, (
        f"CA-CA distance {mean_ca_dist:.2f} outside expected range [3.0, 4.5]"
    )
    print("       ✓ Folded structure looks reasonable!")

    return result


def benchmark_speeds(jax_esm3, torch_esm3, sequence_tokens, n_warmup=2, n_runs=5):
    print("\n[7/7] Benchmarking speeds...")

    jax_seq_tokens = jnp.array(sequence_tokens.numpy())

    print("\n       --- PyTorch (eager, CPU) ---")
    for _ in range(n_warmup):
        with torch.no_grad():
            torch_esm3.forward(sequence_tokens=sequence_tokens.unsqueeze(0))

    torch_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            torch_esm3.forward(sequence_tokens=sequence_tokens.unsqueeze(0))
        torch_times.append(time.perf_counter() - t0)

    torch_mean = np.mean(torch_times)
    torch_std = np.std(torch_times)
    print(f"       {torch_mean:.3f}s ± {torch_std:.3f}s per forward pass")

    print("\n       --- JAX (no JIT) ---")
    for _ in range(n_warmup):
        out = jax_esm3(sequence_tokens=jax_seq_tokens)
        jax.block_until_ready(out)

    jax_nojit_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = jax_esm3(sequence_tokens=jax_seq_tokens)
        jax.block_until_ready(out)
        jax_nojit_times.append(time.perf_counter() - t0)

    jax_nojit_mean = np.mean(jax_nojit_times)
    jax_nojit_std = np.std(jax_nojit_times)
    print(f"       {jax_nojit_mean:.3f}s ± {jax_nojit_std:.3f}s per forward pass")

    print("\n       --- JAX (JIT compilation) ---")

    t0 = time.perf_counter()
    jit_out = jit_forward(jax_esm3, jax_seq_tokens)
    jax.block_until_ready(jit_out)
    compile_time = time.perf_counter() - t0
    print(f"       Compilation time: {compile_time:.3f}s")

    print("\n       --- JAX (JIT, post-compilation) ---")
    for _ in range(n_warmup):
        out = jit_forward(jax_esm3, jax_seq_tokens)
        jax.block_until_ready(out)

    jax_jit_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = jit_forward(jax_esm3, jax_seq_tokens)
        jax.block_until_ready(out)
        jax_jit_times.append(time.perf_counter() - t0)

    jax_jit_mean = np.mean(jax_jit_times)
    jax_jit_std = np.std(jax_jit_times)
    print(f"       {jax_jit_mean:.3f}s ± {jax_jit_std:.3f}s per forward pass")

    print("\n       --- Summary ---")
    print(f"       PyTorch (eager):    {torch_mean:.3f}s")
    print(f"       JAX (no JIT):       {jax_nojit_mean:.3f}s")
    print(f"       JAX (JIT compile):  {compile_time:.3f}s (one-time cost)")
    print(f"       JAX (JIT cached):   {jax_jit_mean:.3f}s")
    if jax_jit_mean > 0:
        print(f"       JIT speedup vs PyTorch: {torch_mean / jax_jit_mean:.2f}x")


def main():
    print("=" * 70)
    print("JAX/Equinox ESM3 Protein Folding Example")
    print("=" * 70)

    sequence = "MGSSHHHHHHSSGLVPRGSHMASVQPLASCFSNRYYQLSSNAQRFAGRTYWKATGEDFEYQP"
    print(f"\nTarget sequence ({len(sequence)} residues):")
    print(f"  {sequence}\n")

    jax_esm3, torch_esm3, jax_decoder, torch_decoder = load_models()
    sequence_tokens = encode_sequence(sequence, torch_esm3)

    assert_forward_pass(jax_esm3, torch_esm3, sequence_tokens)
    assert_decoder(jax_decoder, torch_decoder, seq_len=len(sequence) + 2)
    fold_and_validate(jax_esm3, jax_decoder, sequence)
    benchmark_speeds(jax_esm3, torch_esm3, sequence_tokens)

    print("\n" + "=" * 70)
    print("All checks passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
