import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from ECG_Dataset import ECG_Dataset
from draw_utils import LEAD_NAMES
from ecg_deformation_model import ECGDeformationModel, normalize_beat


def _pick_random_record_with_beats(dataset: ECG_Dataset):
    records = [r for r in dataset.records if getattr(r, "beat_representations", None) is not None]
    records = [r for r in records if len(r.beat_representations) > 0]
    if not records:
        raise RuntimeError("No records with beat representations found in dataset.")
    return random.choice(records)


def _finite_or_raise(name: str, arr: np.ndarray):
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"{name}: contains NaN/Inf (first bad index={bad[0].tolist() if len(bad) else 'n/a'})")

def _signal_stats(tag: str, arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f"{tag}: empty")
        return
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    ptp = float(np.ptp(arr))
    print(f"{tag}: shape={arr.shape}, mean={mean:.6g}, std={std:.6g}, ptp={ptp:.6g}")


def _format_patient_age(age):
    try:
        age_f = float(age)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(age_f):
        return "n/a"
    return f"{age_f:.1f}y"


def _extract_state_dict(checkpoint_obj):
    # Plain state_dict saved via torch.save(model.state_dict(), path)
    if isinstance(checkpoint_obj, dict) and checkpoint_obj:
        if all(isinstance(k, str) for k in checkpoint_obj.keys()):
            if all(torch.is_tensor(v) for v in checkpoint_obj.values()):
                return checkpoint_obj

        # Common checkpoint layouts
        for key in ("state_dict", "model_state_dict", "model"):
            maybe_state = checkpoint_obj.get(key)
            if isinstance(maybe_state, dict):
                return maybe_state
    raise TypeError("Unsupported checkpoint format: unable to locate a state_dict.")


def _normalize_state_dict_keys(state_dict):
    normalized = state_dict
    for prefix in ("_orig_mod.", "module."):
        if any(k.startswith(prefix) for k in normalized.keys()):
            normalized = {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in normalized.items()
            }
    return normalized


def _load_model_weights(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _normalize_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with model architecture.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )


def _draw_signal_overlay(signals, sampling_rate, subtitle, leads=None, sharey=True, patient_age=None):
    """
    signals: dict[str, np.ndarray] -> each [T, C]
    """
    if leads is None:
        leads = list(range(signals[next(iter(signals))].shape[1]))

    first = next(iter(signals.values()))
    num_samples = first.shape[0]
    duration_seconds = num_samples / float(sampling_rate)
    time = np.arange(num_samples) / float(sampling_rate)
    num_plots = len(leads)

    is_short_signal = duration_seconds < 2.5
    if is_short_signal:
        cols = 4 if num_plots >= 4 else num_plots
        rows = int(np.ceil(num_plots / cols))
        fig_width = 4 * cols
        fig_height = 3 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=sharey)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]
    else:
        cols = 1
        rows = num_plots
        fig_width = max(12, duration_seconds * 2)
        fig_height = 2.0 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=sharey)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]

    # Reserve blue for "Original"; use a non-blue palette for other series.
    non_original_palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    labels = list(signals.keys())
    if "Original" in labels:
        labels = [l for l in labels if l != "Original"] + ["Original"]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    non_original_labels = [l for l in labels if l != "Original"]
    non_original_color = {
        label: non_original_palette[i % len(non_original_palette)]
        for i, label in enumerate(non_original_labels)
    }
    non_original_style = {
        label: styles[i % len(styles)]
        for i, label in enumerate(non_original_labels)
    }

    for i, ax in enumerate(axes_flat):
        if i >= num_plots:
            ax.axis("off")
            continue

        lead_idx = leads[i]
        for label in labels:
            sig = signals[label]
            if label == "Original":
                color = "#1f77b4"
                lw = 1.6 if is_short_signal else 1.2
                alpha = 1.0
                zorder = 3
                style = "-"
            else:
                color = non_original_color[label]
                lw = 1.1 if is_short_signal else 0.9
                alpha = 0.9
                zorder = 2
                style = non_original_style[label]
            ax.plot(
                time,
                sig[:, lead_idx],
                linewidth=lw,
                color=color,
                alpha=alpha,
                linestyle=style,
                label=label,
                zorder=zorder,
            )

        lead_title = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f"Lead {lead_idx}"
        ax.set_title(lead_title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        if is_short_signal:
            if i % cols == 0:
                ax.set_ylabel("mV", fontsize=9)
            if i >= (rows - 1) * cols:
                ax.set_xlabel("Time (s)", fontsize=9)
        else:
            ax.set_ylabel("mV", fontsize=9)
            if i == num_plots - 1:
                ax.set_xlabel("Time (s)", fontsize=10)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    patient_age_text = _format_patient_age(patient_age)
    plt.suptitle(
        f"{subtitle} ({duration_seconds:.2f}s) | Patient age={patient_age_text}",
        fontsize=14,
        fontweight="bold",
        y=0.98 if is_short_signal else 1.005,
    )
    plt.tight_layout()
    return fig


def _autoscale_y_for_small_diffs(fig, signals, leads, q=0.995):
    """
    For diff plots, tighten y-limits per-axis to make small deformations visible.
    Uses robust percentile of absolute values across all signals for the lead.
    """
    axes = fig.get_axes()
    for i, lead_idx in enumerate(leads):
        if i >= len(axes):
            break
        ax = axes[i]
        vals = []
        for sig in signals.values():
            vals.append(sig[:, lead_idx])
        v = np.concatenate(vals, axis=0)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        lim = float(np.quantile(np.abs(v), q))
        lim = max(lim, 1e-6)
        ax.set_ylim(-lim, lim)


def _pick_beat_and_mask(record, beat_idx: int):
    beats = getattr(record, "beat_representations", None)
    if beats is None or len(beats) == 0:
        raise ValueError("Record has no beat representations.")

    idx = max(0, min(int(beat_idx), len(beats) - 1))
    beat = np.asarray(beats[idx], dtype=np.float32)
    if beat.ndim != 2:
        raise ValueError(f"Expected beat as 2D array, got shape={beat.shape}")

    expected_leads = len(LEAD_NAMES)
    # Stored beat layout is expected [T, C], but support [C, T] for robustness.
    if beat.shape[1] == expected_leads:
        beat_tc = beat
    elif beat.shape[0] == expected_leads:
        beat_tc = beat.T
    else:
        raise ValueError(
            f"Unable to infer beat layout for shape={beat.shape}. "
            f"Expected one axis to be {expected_leads} leads."
        )

    mask_t = None
    masks = getattr(record, "beat_masks", None)
    if masks is not None and len(masks) > idx:
        mask_t = np.asarray(masks[idx], dtype=np.float32).reshape(-1)
        if mask_t.shape[0] != beat_tc.shape[0]:
            mask_t = None

    return idx, beat_tc, mask_t


def _to_np_tc(x: torch.Tensor) -> np.ndarray:
    # Accept [C,T] or [1,C,T] -> return [T,C]
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected tensor with shape [C,T] or [1,C,T], got {tuple(x.shape)}")
    return x.detach().cpu().numpy().transpose(1, 0)


def main():
    ################### CONFIG ###################
    model_path = "models/ecg_def_mod100.pth"
    dataset_path = "./data/ptbxl_100_T.pkl"
    out_dir = "./recon"
    target_ages = [20, 50, 80]
    num_examples = 2
    beat_idx_default = 2
    PLOT_LEADS = [0, 1, 6]  # e.g. [0] for only Lead I
    FS_FALLBACK = 100


    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset from {dataset_path} ...")
    ds = ECG_Dataset.load(dataset_path)

    print(f"Loading model from {model_path} ...")
    model = ECGDeformationModel().to(device)
    _load_model_weights(model, model_path, device)
    model.eval()

    neutral_age = float(getattr(model, "NEUTRAL_AGE", 50.0))
    target_ages = sorted(set(float(a) for a in target_ages) | {neutral_age})

    for i in range(int(num_examples)):
        record = _pick_random_record_with_beats(ds)
        record_id = getattr(record, "id", f"rec{i}")
        fs = float(getattr(record, "fs", FS_FALLBACK))

        beat_idx = beat_idx_default
        beat_idx, beat, beat_mask = _pick_beat_and_mask(record, beat_idx)

        beat_t = torch.from_numpy(beat).permute(1, 0).unsqueeze(0).to(device)  # [1, C, T]
        if beat_mask is not None:
            beat_mask_t = torch.from_numpy(beat_mask).unsqueeze(0).to(device)
            beat_norm = normalize_beat(beat_t, beat_mask_t)
        else:
            beat_norm = normalize_beat(beat_t)

        with torch.no_grad():
            # canonical path
            z = model.encode(beat_norm)
            x50 = model.canonical(z)
            # Get normalized axis direction B_dir for this sample.
            if hasattr(model, "_sensitivity_dir_field"):
                b_dir = model._sensitivity_dir_field(z)
            else:
                B_raw = model.sensitivity(z)  # [1, C, T]
                b_dir = B_raw / (B_raw.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-6).view(-1, 1, 1))

            x50_np = _to_np_tc(x50)

            recon_by_age = {}
            diff_by_age = {}
            age_pred_by_age = {}
            g_age_by_age = {}
            g_hat_by_age = {}
            delta_norm_by_age = {}

            for age in target_ages:
                age_t = torch.tensor([float(age)], device=device, dtype=beat_norm.dtype)
                delta_age = model.deformation(z, age_t)  # [1, C, T]
                x_recon_age = x50 + delta_age

                recon_by_age[age] = _to_np_tc(x_recon_age)
                diff_by_age[age] = _to_np_tc(delta_age)

                age_pred_by_age[age] = float(model.predict_age(x_recon_age).item())
                g_age_by_age[age] = float(model.age_scalar(age_t).item())
                g_hat_by_age[age] = float(model._axis_projection(delta_age, b_dir).item())
                delta_norm_by_age[age] = float(delta_age.flatten(1).norm(dim=1).item())

            # reconstruction at true age
            age_true = float(getattr(record, "age", neutral_age))
            age_true_t = torch.tensor([age_true], device=device, dtype=beat_norm.dtype)
            x_recon_true = x50 + model.deformation(z, age_true_t)
            rec_l1 = float(torch.mean(torch.abs(x_recon_true - beat_norm)).item())
            rec_mse = float(torch.mean((x_recon_true - beat_norm) ** 2).item())

            # observed projection
            delta_obs = beat_norm - x50
            g_hat_obs = float(model._axis_projection(delta_obs, b_dir).item())
            age_pred_obs = float(model.predict_age_center(torch.tensor([g_hat_obs], device=device)).item())
            age_pred_direct = float(model.predict_age(beat_norm).item())

            b_dir_norm = float(b_dir.flatten(1).norm(dim=1).item())

            # x50 leakage-like stats (single beat; not a dataset test, but good sanity)
            mean_abs_x50 = float(x50.abs().mean().item())
            std_x50 = float(x50.std().item())

        orig = _to_np_tc(beat_norm)

        # Basic validity checks
        _finite_or_raise("Original (beat_norm)", orig)
        _finite_or_raise("x50", x50_np)
        for age in target_ages:
            _finite_or_raise(f"x_recon({age})", recon_by_age[age])
            _finite_or_raise(f"delta({age})", diff_by_age[age])

        # Print compact diagnostics
        print("\n" + "-" * 80)
        print(f"[{i+1}/{num_examples}] Record id={record_id} | beat={beat_idx} | fs={fs:.1f} | age={_format_patient_age(getattr(record, 'age', None))}")
        print(f"Axis: ||B_dir||={b_dir_norm:.6f} | g_hat_obs={g_hat_obs:.6g} | age_pred_center(g_hat_obs)={age_pred_obs:.3f} | age_pred_direct(beat)={age_pred_direct:.3f}")
        print(f"Recon@true_age: L1={rec_l1:.8g} | MSE={rec_mse:.8g}")
        print(f"x50 stats: mean|x50|={mean_abs_x50:.6g} | std(x50)={std_x50:.6g}")
        for age in target_ages:
            mag = float(np.mean(np.abs(diff_by_age[age])))
            signed_mean = float(np.mean(diff_by_age[age]))
            print(
                f"  age={age:5.1f}: |delta|={mag:.8g}, signed_mean={signed_mean:.8g}, "
                f"||delta||_2={delta_norm_by_age[age]:.8g}, g(age)={g_age_by_age[age]:.8g}, "
                f"g_hat(delta)={g_hat_by_age[age]:.8g}, age_pred(x_recon)={age_pred_by_age[age]:.3f}"
            )

        # Plot 1: absolute reconstructions
        abs_signals = {"Original": orig}
        for age in target_ages:
            abs_signals[f"Age {age:g}"] = recon_by_age[age]

        fig_abs = _draw_signal_overlay(
            abs_signals,
            sampling_rate=fs,
            subtitle=f"Absolute reconstructions (v3 axis) | Record id={record_id}, beat={beat_idx}",
            leads=PLOT_LEADS,
            sharey=True,
            patient_age=getattr(record, "age", None),
        )
        out_path_abs = os.path.join(out_dir, f"dv_{record_id}_beat{beat_idx}.png")
        plt.savefig(out_path_abs, dpi=150, bbox_inches="tight")
        plt.close(fig_abs)
        print(f"Saved: {out_path_abs}")

        # Plot 2: delta(age) relative to neutral (include neutral as ~0 for reference)
        diff_signals = {}
        for age in target_ages:
            if abs(float(age) - neutral_age) < 1e-6:
                continue
            diff_signals[f"Age {age:g} - {neutral_age:g}"] = diff_by_age[age]
        diff_signals[f"Age {neutral_age:g} - {neutral_age:g}"] = (
            diff_by_age[neutral_age]
            if neutral_age in diff_by_age
            else np.zeros_like(x50_np)
        )


        fig_diff = _draw_signal_overlay(
            diff_signals,
            sampling_rate=fs,
            subtitle=f"Deformation Delta(z, age) | Record id={record_id}, beat={beat_idx}",
            leads=PLOT_LEADS,
            sharey=False,
            patient_age=getattr(record, "age", None),
        )
        _autoscale_y_for_small_diffs(fig_diff, diff_signals, leads=PLOT_LEADS, q=0.995)
        out_path_diff = os.path.join(out_dir, f"dv_diff_{record_id}_beat{beat_idx}.png")
        plt.savefig(out_path_diff, dpi=150, bbox_inches="tight")
        plt.close(fig_diff)
        print(f"Saved: {out_path_diff}")


if __name__ == "__main__":
    main()
