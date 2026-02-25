"""Beat-level data batcher for ECG records.

Extracted from train_deformation.py so it can be shared by the training
loop and the validation suite without circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ECG_Dataset import ECG_Dataset


@dataclass
class BeatBatch:
    beats: torch.Tensor     # [B, C, T]
    ages: torch.Tensor      # [B]
    masks: torch.Tensor     # [B, T]


class ECGBeatBatcher:
    """
    Deterministic beat-level batcher for exactly one pass over a split.

    - __len__ matches the actual number of beat-batches.
    - Uses all beats from all records in the split (or a capped count per record).
    - Shuffles only when requested (train).
    """

    def __init__(
        self,
        ds: ECG_Dataset,
        split: str,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        device: torch.device | None = None,
        pin_memory: bool = True,
        cache_on_device: bool = True,
        cache_safety_margin_gb: float = 1.0,
        max_beats_per_record: int | None = None,
    ):
        self.ds = ds
        self.split = split
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pin_memory = bool(pin_memory) and (self.device.type == "cuda")
        self.cache_on_device = bool(cache_on_device) and (self.device.type == "cuda")
        self.cache_safety_margin_gb = float(cache_safety_margin_gb)
        self.cached_on_device = False
        self.max_beats_per_record = (
            None if max_beats_per_record is None else int(max_beats_per_record)
        )
        if self.max_beats_per_record is not None and self.max_beats_per_record <= 0:
            raise ValueError("max_beats_per_record must be > 0 or None")

        self.records = self.ds.get_data(split)
        if len(self.records) == 0:
            raise ValueError(f"No records found for split={split}")

        r0 = self.records[0].beat_representations
        _, T, C = r0.shape
        self.C = int(C)
        self.T = int(T)

        beat_chunks = []
        age_chunks = []
        mask_chunks = []
        for r in self.records:
            br = r.beat_representations
            if br is None or br.shape[0] == 0:
                continue
            if self.max_beats_per_record is not None:
                br = br[:self.max_beats_per_record]
            m = r.beat_masks
            if m is None or m.shape[0] < br.shape[0]:
                m = (np.abs(br).sum(axis=2) > 0).astype(np.float32)
            else:
                m = m[:br.shape[0]].astype(np.float32, copy=False)
            beat_chunks.append(br.transpose(0, 2, 1).astype(np.float32, copy=False))
            age_chunks.append(np.full((br.shape[0],), float(r.age), dtype=np.float32))
            mask_chunks.append(m)

        if not beat_chunks:
            raise ValueError(f"No beats found for split={split}")

        beats_np = np.concatenate(beat_chunks, axis=0)
        ages_np = np.concatenate(age_chunks, axis=0)
        masks_np = np.concatenate(mask_chunks, axis=0)
        self.total_beats = int(beats_np.shape[0])

        self._beats_all = torch.from_numpy(beats_np)
        self._ages_all = torch.from_numpy(ages_np)
        self._masks_all = torch.from_numpy(masks_np)
        self.cache_bytes = (
            self._nbytes(self._beats_all)
            + self._nbytes(self._ages_all)
            + self._nbytes(self._masks_all)
        )
        if self.pin_memory:
            self._beats_all = self._beats_all.pin_memory()
            self._ages_all = self._ages_all.pin_memory()
            self._masks_all = self._masks_all.pin_memory()

        if self.cache_on_device:
            self._try_cache_on_device()

    def __len__(self) -> int:
        return int(np.ceil(self.total_beats / self.batch_size))

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device, non_blocking=self.pin_memory)

    @staticmethod
    def _nbytes(x: torch.Tensor) -> int:
        return int(x.numel() * x.element_size())

    def _try_cache_on_device(self) -> None:
        required = (
            self._nbytes(self._beats_all)
            + self._nbytes(self._ages_all)
            + self._nbytes(self._masks_all)
        )
        margin = int(max(0.0, self.cache_safety_margin_gb) * (1024 ** 3))

        try:
            with torch.cuda.device(self.device):
                free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes < (required + margin):
                return

            self._beats_all = self._beats_all.to(self.device, non_blocking=self.pin_memory)
            self._ages_all = self._ages_all.to(self.device, non_blocking=self.pin_memory)
            self._masks_all = self._masks_all.to(self.device, non_blocking=self.pin_memory)
            self.pin_memory = False
            self.cached_on_device = True
        except RuntimeError:
            self.cached_on_device = False

    def __iter__(self):
        Bsz = self.batch_size

        if self.cached_on_device:
            if self.shuffle:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(self.seed)
                idxs = torch.randperm(self.total_beats, device=self.device, generator=gen)
            else:
                idxs = torch.arange(self.total_beats, device=self.device)

            for s in range(0, self.total_beats, Bsz):
                idx_t = idxs[s:s + Bsz]
                beats = self._beats_all.index_select(0, idx_t)
                ages = self._ages_all.index_select(0, idx_t)
                masks = self._masks_all.index_select(0, idx_t)
                yield BeatBatch(beats=beats, ages=ages, masks=masks)
            return

        rng = np.random.default_rng(self.seed)
        idxs = np.arange(self.total_beats, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(idxs)

        for s in range(0, len(idxs), Bsz):
            batch_idx = idxs[s:s + Bsz]
            idx_t = torch.from_numpy(batch_idx).long()
            beats = self._beats_all.index_select(0, idx_t)
            ages = self._ages_all.index_select(0, idx_t)
            masks = self._masks_all.index_select(0, idx_t)

            yield BeatBatch(
                beats=self._to_device(beats),
                ages=self._to_device(ages),
                masks=self._to_device(masks),
            )
