import os
import ast
import pickle
import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import LabelEncoder

from ECG_Record import ECG_Record


class ECG_Dataset:
    """
    Manages a collection of ECG_Records.
    Handles file I/O, importing from PTB-XL, and exporting to Arrays.
    """

    def __init__(self, fs=100):
        self.records = []
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.fs = fs

    def __len__(self):
        """Return total number of records in the dataset."""
        return len(self.records)

    def __iter__(self):
        """Iterate over ECG records."""
        return iter(self.records)

    def __getitem__(self, idx):
        """Index access: ds[idx] -> ECG_Record."""
        return self.records[idx]

    def add_record(self, record):
        self.records.append(record)

    def save(self, path):
        """Save the entire dataset object to disk."""
        print(f"Saving dataset to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print("[v] Save complete.")

    @classmethod
    def load(cls, path):
        """Load a dataset object from disk."""
        print(f"Loading dataset from {path}...")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def import_record(self, row, from_path, is_valid=True, reason=None):
        # Load signal
        sig, _ = wfdb.rdsamp(from_path)
        if np.isnan(sig).any():
            sig = np.nan_to_num(sig)

        extra_labels = row.get('extra_labels', [])
        if isinstance(extra_labels, float) and np.isnan(extra_labels):
            extra_labels = []

        # Create Record
        record = ECG_Record(
            id=row['ecg_id'],
            signal=sig,
            age=row['age'],
            sex=row['sex'],
            label=row['diagnostic_class'],
            extra_labels=extra_labels,
            fold=row['strat_fold'],
            fs=self.fs,
            is_valid=is_valid,
            reason=reason
        )

        if record.age > 100:
            record.is_valid = False
            record.reason = "Age > 100"


        record.detect_peaks(0)
        valid_seg = record.segment_beats(target_length=int(1.0 * self.fs))
        if valid_seg:
            # record.remove_base_shift_per_beat()
            record.normalize()
            record.compute_rr()
        else:
            record.is_valid = False
            record.reason = "Segmentation failed"

        return record

    def import_ptbxl(self, base_path, verbose=True):
        """
        Imports data from raw PTB-XL files.
        Performs Filtering -> Loading -> Normalization -> Segmentation.
        """
        print(f"Starting PTB-XL Import from {base_path}...")

        # 1. Load Metadata
        df = pd.read_csv(os.path.join(base_path, "ptbxl_database.csv"))
        df_scps = pd.read_csv(os.path.join(base_path, "scp_statements.csv"), index_col=0)
        df_scps = df_scps[df_scps["diagnostic"] == 1]

        # 2. Clean Metadata (Age, Sex)
        df = df.dropna(subset=["age", "sex"])

        # 3. Parse Labels
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
        df[["diagnostic_class", "extra_labels"]] = df["scp_codes"].apply(
            lambda x: pd.Series(self._pick_subclass(x, df_scps))
        )
        # 4. Iterate and Process
        filename_col = "filename_lr" if self.fs == 100 else "filename_hr"
        noise_cols = ["static_noise", "burst_noise", "electrodes_problems"]
        existing_cols = [c for c in noise_cols if c in df.columns]
        count = 0

        for idx, row in df.iterrows():
            f_path = os.path.join(base_path, row[filename_col])

            # Check noise problems
            is_valid = True
            reason = None
            if existing_cols:
                noise_info = row[existing_cols]
                if noise_info.notnull().any():
                    is_valid = False
                    reasons = []
                    for col in existing_cols:
                        if pd.notnull(row[col]):
                            reasons.append(f"{col}: {row[col]}")
                    reason = "; ".join(reasons)

            record = self.import_record(row, from_path=f_path, is_valid=is_valid, reason=reason)
            if record is not None:
                self.records.append(record)
                count += 1
            if verbose and count % 1000 == 0:
                print(f"Processed {count} records...")

        # 5. Remove all labels with less than 1% of samples
        total_records = len(self.records)
        if total_records == 0:
            print("No valid records imported from PTB-XL.")
            return

        labels_all = np.array([r.label for r in self.records])
        unique_labels, counts = np.unique(labels_all, return_counts=True)
        freqs = counts / float(total_records)

        min_fraction = 0.0029  # 0.29%
        keep_mask = freqs >= min_fraction
        kept_labels = set(unique_labels[keep_mask])
        removed_labels = set(unique_labels[~keep_mask])

        if removed_labels:
            print("\nPruning rare classes (< 1% of samples overall):")
            for lbl, cnt, freq in zip(unique_labels, counts, freqs):
                if lbl in removed_labels:
                    print(f"  - {lbl}: {cnt} samples ({freq * 100:.3f}%)  -> REMOVED")

            self.records = [r for r in self.records if r.label in kept_labels]
            print(f"After pruning: {len(self.records)} records remain.\n")
        else:
            print("\nNo classes below 1% threshold. Keeping all records.\n")

        # Fit Label Encoder on the loaded data (after removing rare classes)
        all_labels = [r.label for r in self.records]
        self.label_encoder.fit(all_labels)
        self.class_names = list(self.label_encoder.classes_)

        print(f"Import Finished. Total valid records: {len(self.records)}")

    def get_data(self, split='all'):
        """
        Returns (X, y, age, sex) arrays for a specific split (train/val/test).
        splits: 'train' (folds 1-8), 'val' (fold 9), 'test' (fold 10), or 'all'.
        In 'test' split, all records are returned. In other splits, only valid records.
        """
        if not self.records:
            raise ValueError("Dataset is empty.")

        # Define split filter
        if split == 'train':
            filtered = [r for r in self.records if r.fold <= 8]
        elif split == 'val':
            filtered = [r for r in self.records if r.fold == 9]
        elif split == 'test':
            filtered = [r for r in self.records if r.fold == 10]
        else:
            filtered = self.records

        # Filter by validity, except for 'test' split.
        if split != 'test':
            filtered = [r for r in filtered if r.is_valid]
        else:
            filtered = [r for r in filtered if r.is_valid or r.reason != "Age > 100"]

        if not filtered:
            raise ValueError(f"No records found for split '{split}'")

        return filtered

    def get_record(self, idx):
        return self.records[idx]

    def get_record_by_id(self, idx):
        for record in self.records:
            if record.id == idx:
                return record
        return None

    def _aggregate_class(self, scp_dict, scp_df):
        """Helper to map SCP codes to superclass."""
        labels = []
        for code in scp_dict.keys():
            if code in scp_df.index:
                labels.append(scp_df.loc[code, "diagnostic_class"])
        # Deterministic sort
        labels = sorted(list(set(labels)))
        return labels[0] if labels else "NORM"

    def _normalize_subclass(self, subclass):
        if subclass in {"ISCA", "ISCI", "ISC_"}:
            return "ISC"
        if subclass == "LMI":
            return "IMI"
        return subclass

    def _pick_subclass(self, scp_dict, scp_df):
        """
        Selects a single diagnostic subclass:
        - look at weights in scp_dict (dict values);
        - take the subclass with the maximum weight;
        - if weights tie, choose alphabetically.
        If there are no diagnostic codes, return "NORM".
        """
        candidates = []
        total_weight = 0.0

        for code, weight in scp_dict.items():
            if code not in scp_df.index:
                continue
            subclass = scp_df.loc[code, "diagnostic_subclass"]
            subclass = self._normalize_subclass(subclass)
            candidates.append((subclass, weight))
            total_weight += float(weight)

        if not candidates:
            return "NORM", []

        best_weight_by_subclass = {}
        for subclass, weight in candidates:
            if subclass not in best_weight_by_subclass:
                best_weight_by_subclass[subclass] = weight
            else:
                best_weight_by_subclass[subclass] = max(
                    best_weight_by_subclass[subclass], weight
                )

        best_subclass = max(
            best_weight_by_subclass.items(),
            key=lambda item: (item[1], item[0])  # (weight, subclass_name)
        )[0]

        extra_labels = []
        if total_weight > 100.0:
            extra_labels = sorted([s for s in best_weight_by_subclass.keys() if s != best_subclass])

        return best_subclass, extra_labels
