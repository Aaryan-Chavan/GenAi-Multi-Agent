# Embeddings/quantization.py

import json
from pathlib import Path

import numpy as np

from Config.settings import EMBEDDINGS_DIR


class EmbeddingQuantizer:
    """
    INT8 per-channel quantization for float32 embedding matrices.

    Per-channel (one scale per dimension) minimises quantization error
    for cosine-similarity search versus a single global scale.

    Compression : float32 (4 B) → int8 (1 B)  ≈ 4×
    Typical quality : cosine_delta < 0.001 for 768-dim+ models
    """

    def __init__(self):
        self.output_dir = Path(EMBEDDINGS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def _validate(embeddings: np.ndarray) -> np.ndarray:
        if embeddings is None:
            raise ValueError("embeddings is None.")
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D array (N, D), got shape {arr.shape}.")
        if arr.size == 0:
            raise ValueError("embeddings array is empty.")
        return arr

    # --------------------------------------------------------
    # QUANTIZE
    # --------------------------------------------------------

    def quantize_int8(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-channel INT8 quantization.

        Returns
        -------
        quantized : int8 ndarray,    shape (N, D)
        scales    : float32 ndarray, shape (D,)
        """
        embeddings = self._validate(embeddings)

        scales = np.max(np.abs(embeddings), axis=0)          # shape (D,)
        scales = np.where(scales == 0.0, 1.0, scales)        # guard dead dims

        quantized = np.round(embeddings / scales * 127.0).astype(np.int8)
        return quantized, scales

    # --------------------------------------------------------
    # DEQUANTIZE
    # --------------------------------------------------------

    def dequantize_int8(
        self, quantized: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        """Reconstruct float32 from int8 + per-channel scales."""
        return quantized.astype(np.float32) / 127.0 * scales

    # --------------------------------------------------------
    # RECONSTRUCTION ERROR
    # --------------------------------------------------------

    def reconstruction_error(
        self,
        original:  np.ndarray,
        quantized: np.ndarray,
        scales:    np.ndarray,
    ) -> dict:
        """
        Quality check before writing to disk.

        Returns MAE and mean row-wise cosine similarity between
        original and dequantized embeddings.
        """
        orig_f32      = np.asarray(original, dtype=np.float32)
        reconstructed = self.dequantize_int8(quantized, scales)

        mae = float(np.mean(np.abs(orig_f32 - reconstructed)))

        eps        = 1e-9
        orig_norm  = orig_f32      / (np.linalg.norm(orig_f32,      axis=1, keepdims=True) + eps)
        recon_norm = reconstructed / (np.linalg.norm(reconstructed, axis=1, keepdims=True) + eps)
        cos_sim    = float(np.mean(np.sum(orig_norm * recon_norm, axis=1)))

        return {
            "mae":                    round(mae, 6),
            "mean_cosine_similarity": round(cos_sim, 6),
            "cosine_delta":           round(1.0 - cos_sim, 6),
        }

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save_quantized(
        self,
        quantized_embeddings: np.ndarray,
        scales:               np.ndarray,
        file_name:            str  = "quantized_embeddings.npz",
        error_metrics:        dict = None,
    ) -> None:
        """
        Save compressed archive + JSON manifest.

        Archive keys
        ------------
        embeddings : int8 vectors,  shape (N, D)
        scales     : float32 scales, shape (D,)
        """
        out_file      = self.output_dir / file_name
        manifest_file = self.output_dir / file_name.replace(".npz", "_manifest.json")

        np.savez_compressed(out_file, embeddings=quantized_embeddings, scales=scales)

        manifest = {
            "quantization_mode": "int8_per_channel",
            "shape":             list(quantized_embeddings.shape),
            "dtype":             str(quantized_embeddings.dtype),
            "scales_shape":      list(scales.shape),
            "file":              out_file.name,
        }
        if error_metrics:
            manifest["reconstruction_error"] = error_metrics

        with open(manifest_file, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        print(f"\n  Quantized -> {out_file}")
        print(f"  Manifest  -> {manifest_file}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load_quantized(
        self, file_name: str = "quantized_embeddings.npz"
    ) -> Tuple[np.ndarray, np.ndarray]:
        path = self.output_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Quantized file not found: {path}")
        data = np.load(path)
        return data["embeddings"], data["scales"]

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def summary(
        self,
        original:      np.ndarray,
        quantized:     np.ndarray,
        error_metrics: dict = None,
    ) -> None:
        orig_mb  = original.nbytes  / 1024 ** 2
        quant_mb = quantized.nbytes / 1024 ** 2
        ratio    = orig_mb / quant_mb if quant_mb > 0 else float("inf")

        print("\n" + "=" * 60)
        print("QUANTIZATION SUMMARY")
        print("=" * 60)
        print(f"Original  shape  : {original.shape}")
        print(f"Quantized shape  : {quantized.shape}")
        print(f"Original  size   : {orig_mb:.2f} MB")
        print(f"Quantized size   : {quant_mb:.2f} MB")
        print(f"Compression      : {ratio:.2f}×")

        if error_metrics:
            print(f"MAE              : {error_metrics['mae']}")
            print(f"Cosine sim       : {error_metrics['mean_cosine_similarity']}")
            print(f"Cosine delta     : {error_metrics['cosine_delta']}")

        print("=" * 60)