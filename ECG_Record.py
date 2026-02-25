import numpy as np
import neurokit2 as nk

from signal_utils import calc_sqi_metrics


class ECG_Record:
    """
    Represents a single ECG record with 12-lead signal and metadata.
    Contains methods for self-normalization and segmentation.
    """

    def __init__(self, id, signal, age, sex, label=None, extra_labels=None, fold=None, fs=100, is_valid=True, reason=None):
        self.id = id
        self.signal = signal.astype(np.float32)  # Shape: [T, C]
        # B - beat
        self.beats = None  # [B, T, C]
        self.beat_representations = None  # [B, T, C]
        self.beat_masks = None  # [B, T]
        self.beat_sigma = None  # [B, C]
        self.beat_amax = None  # [B, C]
        self.rr_intervals = None

        self.age = float(age)
        self.sex = sex  # Simple encoding: Male=0, Female=1
        self.label = label
        self.extra_labels = extra_labels or []
        self.fold = fold

        self.fs = fs

        self.r_peaks = []
        self.r_peak_candidates = []

        self.is_valid = is_valid
        self.reason = reason

    def is_acceptable(self, min_kurtosis=5.0, min_psqi=0.5):
        """Checks if the average lead quality is acceptable."""
        metrics = calc_sqi_metrics(self.signal, self.fs)
        mean_k = np.mean(metrics["kSQI"])
        mean_p = np.mean(metrics["pSQI"])
        if mean_k < min_kurtosis:
            return False
        if mean_p < min_psqi:
            return False
        return True

    def detect_peaks(self, lead_idx=0):
        """
        Detects R-peaks in the ECG signal using the specified lead.
        Then searches all other leads for additional peak candidates that are
        not close to the primary peaks.

        Args:
            lead_idx: Index of the lead to use for primary peak detection (default: 0 for Lead I)

        Returns:
            bool: True if peaks were successfully detected, False otherwise
        """
        # Store candidates with lead info and amplitude for later comparison
        candidate_list = []  # List of (peak_idx, lead_idx, amplitude)
        # Step 1: Detect peaks in primary lead
        try:
            cleaned = nk.ecg_clean(self.signal[:, lead_idx], sampling_rate=self.fs)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
            peaks = info["ECG_R_Peaks"]
            peaks = sorted([int(p) for p in peaks if not np.isnan(p)])
        except Exception:
            peaks = []

        self.r_peaks = np.array(peaks)

        if len(peaks) <= 2:
            return False

        # Step 2: Search for additional peaks in all other leads
        num_leads = self.signal.shape[1]
        threshold = self.fs / 5  # Minimum distance to consider a peak as "not close"

        for lead in range(num_leads):
            if lead == lead_idx:
                continue  # Skip the primary lead

            try:
                cleaned = nk.ecg_clean(self.signal[:, lead], sampling_rate=self.fs)
                _, info = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
                other_peaks = info["ECG_R_Peaks"]
                other_peaks = [int(p) for p in other_peaks if not np.isnan(p)]

                # Check each peak in this lead
                for candidate_peak in other_peaks:
                    # Check if this peak is close to any peak in r_peaks
                    is_close = False
                    for r_peak in self.r_peaks:
                        if abs(candidate_peak - r_peak) < threshold:
                            is_close = True
                            break

                    # If not close to any existing peak, add to candidates with lead info
                    if not is_close:
                        amplitude = abs(self.signal[candidate_peak, lead])
                        candidate_list.append((candidate_peak, lead, amplitude))

            except Exception:
                continue  # Skip this lead if detection fails

        # Step 3: Filter candidates to ensure only one peak within fs/5
        # Sort candidates by position
        candidate_list.sort(key=lambda x: x[0])

        filtered_candidates = []
        i = 0
        while i < len(candidate_list):
            # Find all candidates within threshold of current candidate
            group = [candidate_list[i]]
            j = i + 1
            while j < len(candidate_list) and abs(candidate_list[j][0] - candidate_list[i][0]) < threshold:
                group.append(candidate_list[j])
                j += 1

            # Require detection in at least 6 leads
            unique_leads = {g[1] for g in group}
            if len(unique_leads) >= 6:
                # Select the candidate with maximum amplitude
                best_candidate = max(group, key=lambda x: x[2])
                filtered_candidates.append(best_candidate[0])

            # Move to next group
            i = j

        self.r_peak_candidates = np.array(filtered_candidates)

        # Step 4: Add filtered candidates to r_peaks
        # Conditions:
        #  - amplitude > 0 (positive polarity in the primary lead)
        #  - amplitude is at least 40% of BOTH neighboring R-peaks (in primary lead)
        cand = np.array(filtered_candidates, dtype=int)
        if cand.size > 0 and self.r_peaks.size >= 3:
            base_peaks = np.array(self.r_peaks, dtype=int)
            base_peaks.sort()

            cand = np.unique(cand)
            cand.sort()

            accepted = []
            for p in cand:
                # amplitude > 0 in the chosen reference lead
                a = float(self.signal[p, lead_idx])
                if a < -0.1:
                    continue

                # need two neighbors among existing r_peaks
                pos = int(np.searchsorted(base_peaks, p))
                if pos >= len(base_peaks):
                    continue

                prev_p = int(base_peaks[pos - 1])
                next_p = int(base_peaks[pos])

                a_prev = float(abs(self.signal[prev_p, lead_idx]))
                a_next = float(abs(self.signal[next_p, lead_idx]))
                a_abs = float(abs(a))

                if a_abs >= 0.4 * a_prev and a_abs >= 0.4 * a_next:
                    accepted.append(int(p))

            if accepted:
                self.r_peaks = np.unique(np.concatenate([base_peaks, np.array(accepted, dtype=int)]))
                self.r_peaks.sort()

        # Step X: Remove "weak" peaks from r_peaks:
        # if amplitude at peak is < 40% of BOTH neighbors, move it to candidates
        peaks_arr = self.r_peaks
        amps = np.abs(self.signal[peaks_arr, lead_idx]).astype(np.float32)

        keep_mask = np.ones(len(peaks_arr), dtype=bool)
        weak_peaks = []
        weak_threshold = 0.2
        weak_thr_ratio = 0.25

        # do not touch first/last since they don't have two neighbors
        for i in range(1, len(peaks_arr) - 1):
            a = float(amps[i])
            a_prev = float(abs(amps[i - 1]))
            a_next = float(abs(amps[i + 1]))

            if (a < weak_thr_ratio * a_prev) and (a_prev > 0.1) and (a < weak_threshold) and (a_next > 0.1) and (a < weak_thr_ratio * a_next):
                keep_mask[i] = False
                weak_peaks.append((int(peaks_arr[i]), lead_idx, a))

        if weak_peaks:
            # remove from primary peaks
            peaks_arr = peaks_arr[keep_mask]
            self.r_peaks = peaks_arr
            # and add to candidates (so they can be re-considered later)
            candidate_list.extend(weak_peaks)

        return True

    def segment_beats(self, target_length=120):
        """
        Extracts all beats centered on R-peaks.
        Aligns them to a fixed length of 200 (2.0 seconds).

        Note: Requires detect_peaks() to be called first.

        Args:
            target_length: Fixed length of each beat segment in samples (default: 200)

        Returns:
            bool: True if beats were successfully segmented, False otherwise
        """
        if len(self.r_peaks) <= 2:
            return False

        peaks = self.r_peaks.tolist()
        segments = []
        masks = []
        rr_intervals = []
        half_len = target_length // 2
        len_left = half_len
        len_right = target_length - len_left
        signal_len = self.signal.shape[0]
        num_peaks = len(peaks)

        for i in range(1, num_peaks - 1):
            p = peaks[i]
            # 1. Compute RR interval for this beat
            rr_current = (peaks[i] - peaks[i - 1]) / self.fs  # RR_t in seconds
            rr_intervals.append(rr_current)

            # 2. Define the fixed extraction window
            window_start = p - len_left
            window_end = p + len_right

            # 3. Extract with padding if necessary
            if window_start >= 0 and window_end <= signal_len:
                seg = self.signal[window_start:window_end]
            else:
                pad_l = max(0, -window_start)
                pad_r = max(0, window_end - signal_len)
                slice_start = max(0, window_start)
                slice_end = min(signal_len, window_end)
                seg_part = self.signal[slice_start:slice_end]
                seg = np.pad(seg_part, ((pad_l, pad_r), (0, 0)), mode='constant', constant_values=0)

            # Make a writable copy for masking
            seg = seg.copy()

            # 4. Calculate Natural Boundaries (midpoints to neighbors)
            prev_p = peaks[i - 1]
            natural_start = (p + prev_p) // 2

            if i == num_peaks - 1:
                natural_end = signal_len
            else:
                next_p = peaks[i + 1]
                natural_end = (p + next_p) // 2

            # 5. Apply Mask (Visual Isolation)
            rel_nat_start = natural_start - window_start
            rel_nat_end = natural_end - window_start

            # Zero out anything before the natural start
            if rel_nat_start > 0:
                cut_idx = int(max(0, rel_nat_start))
                if cut_idx < target_length:
                    seg[:cut_idx, :] = 0

            # Zero out anything after the natural end
            if rel_nat_end < target_length:
                cut_idx = int(min(target_length, rel_nat_end))
                if cut_idx > 0:
                    seg[cut_idx:, :] = 0

            start = int(np.clip(rel_nat_start, 0, target_length))
            end = int(np.clip(rel_nat_end, 0, target_length))
            m = np.zeros((target_length,), dtype=np.float32)
            if end > start:
                m[start:end] = 1.0

            masks.append(m)
            segments.append(seg)

        if segments:
            self.beats = np.stack(segments, axis=0)
            self.beat_masks = np.stack(masks, axis=0)
            self.rr_intervals = np.array(rr_intervals, dtype=np.float32)
            return True
        else:
            return False

    def remove_base_shift_per_beat(self):
        """
        Per-beat, per-lead base shift removal using the masked region:
        - If all masked samples > 0: subtract masked min (bring min to 0).
        - If all masked samples < 0: subtract masked max (bring max to 0).
        - Otherwise: no shift.
        Updates self.beats in-place and stores shift in self.beat_shift.
        """
        if self.beats is None or self.beat_masks is None:
            return self

        beats = self.beats.astype(np.float32)  # [B, T, C]
        masks = self.beat_masks.astype(np.float32)  # [B, T]
        m = masks[:, :, None]  # [B, T, 1]

        # Consider only masked samples when computing min/max
        beats_masked = np.where(m > 0, beats, np.nan)
        bmin = np.nanmin(beats_masked, axis=1)  # [B, C]
        bmax = np.nanmax(beats_masked, axis=1)  # [B, C]

        shift = np.zeros_like(bmin, dtype=np.float32)
        shift = np.where(bmin > 0, bmin, shift)  # all-positive => subtract min
        shift = np.where(bmax < 0, bmax, shift)  # all-negative => subtract max

        beats = beats - shift[:, None, :]  # apply shift to full beat window
        beats = beats * m  # keep outside-mask region at 0

        self.beats = beats
        return self

    def normalize(self):
        """Per-Record / Per-Lead Normalization using mask."""

        beats = self.beats.astype(np.float32)  # [B, T, C]
        masks = self.beat_masks.astype(np.float32)  # [B, T]
        m = masks[:, :, None]  # [B, T, 1]

        denom = np.sum(m, axis=1, keepdims=True)  # [B, 1, 1]
        denom = np.maximum(denom, 1.0)

        mean = np.sum(beats * m, axis=1, keepdims=True) / denom
        var = np.sum(((beats - mean) ** 2) * m, axis=1, keepdims=True) / denom
        std = np.sqrt(np.maximum(var, 1e-6))  # [B, 1, C]

        self.beat_sigma = std[:, 0, :]  # [B, C]
        self.beat_amax = np.max(np.abs(beats * m), axis=1)  # [B, C]

        rep = (beats - mean) / std
        rep = rep * m  # keep masked-out region at 0
        self.beat_representations = rep  # [B, T, C]
        return self

    def compute_rr(self):
        """
        Compute RR interval features for attention mechanism:
        - rr_norm: RR_t / mean_RR_record
        - rr_diff: RR_t - RR_{t-1}
        - rr_z: (RR_t - mean_RR_record) / std_RR_record
        """
        rr = self.rr_intervals  # [B]
        B = len(rr)
        # Compute record-level statistics
        mean_rr = np.mean(rr)
        std_rr = np.std(rr)
        std_rr = max(std_rr, 1e-6)  # Avoid division by zero
        # Component 1: Normalized RR
        rr_norm = rr / (mean_rr + 1e-6)  # [B]
        # Component 2: RR difference (first beat gets 0)
        rr_diff = np.zeros(B, dtype=np.float32)
        if B > 1:
            rr_diff[1:] = rr[1:] - rr[:-1]
        # Component 3: Z-scored RR
        rr_z = (rr - mean_rr) / std_rr  # [B]
        # Stack features: [B, 3]
        self.rr_features = np.stack([rr_norm, rr_diff, rr_z], axis=1)

    def get_sex(self):
        return 'M' if self.sex == 0 else 'F'

    def __repr__(self):
        return f"ECG_Record(id={self.id}, age={self.age}, sex={self.get_sex()}, label={self.label})"
