import numpy as np
import math
from matplotlib import pyplot as plt

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def draw_signal_reconstruction_figure(original, recon, sampling_rate, subtitle, leads=None):
    """
    Returns a Matplotlib figure that overlays original vs reconstructed signal for each lead.
    original, recon: [T, C]
    """
    if leads is None:
        leads = list(range(original.shape[1]))

    num_samples = original.shape[0]
    duration_seconds = num_samples / sampling_rate
    t = np.arange(num_samples) / sampling_rate
    num_plots = len(leads)

    is_short = duration_seconds < 2.5
    if is_short:
        cols = 4 if num_plots >= 4 else num_plots
        rows = math.ceil(num_plots / cols)
        fig_w, fig_h = 4 * cols, 3 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]
    else:
        cols, rows = 1, num_plots
        fig_w = max(12, duration_seconds * 2)
        fig_h = 2.0 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharex=True)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        if i >= num_plots:
            ax.axis('off')
            continue

        lead_idx = leads[i]
        ax.plot(t, original[:, lead_idx], linewidth=1.2, color='#1f77b4', label='Original')
        ax.plot(t, recon[:, lead_idx], linewidth=1.2, color='#ff7f0e', alpha=0.9, label='Reconstruction')
        ax.set_title(f'{LEAD_NAMES[lead_idx]}', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)

        if is_short:
            if i % cols == 0:
                ax.set_ylabel('mV', fontsize=9)
            if i >= (rows - 1) * cols:
                ax.set_xlabel('Time (s)', fontsize=9)
        else:
            ax.set_ylabel('mV', fontsize=9)
            if i == num_plots - 1:
                ax.set_xlabel('Time (s)', fontsize=10)

        if i == 0:
            ax.legend(loc='upper right', fontsize=8, frameon=True)

    fig.suptitle(f"{subtitle} ({duration_seconds:.2f}s)", fontsize=14, fontweight='bold',
                 y=0.98 if is_short else 1.005)
    fig.tight_layout()
    return fig


def draw_som_location_figure(som_dim, rc, subtitle="SOM location"):
    """
    som_dim: (H, W)
    rc: (row, col)
    """
    H, W = som_dim
    r, c = rc

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.zeros((H, W), dtype=np.float32), cmap='Greys', vmin=0.0, vmax=1.0)

    # grid lines
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)

    # marker (col on x, row on y)
    ax.scatter([c], [r], s=250, marker='x', color='red', linewidths=3)

    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.set_title(subtitle, fontsize=12, fontweight='bold')
    return fig


def draw_signal(signal, sampling_rate, subtitle, leads=None):
    if leads is None:
        leads = list(range(12))

    num_samples = signal.shape[0]
    duration_seconds = num_samples / sampling_rate
    time = np.arange(num_samples) / sampling_rate
    num_plots = len(leads)

    # LAYOUT LOGIC
    # If signal is short (< 2.5s), use a GRID layout to save vertical space.
    # If signal is long (>= 2.5s), use a VERTICAL STACK to maximize width.
    is_short_signal = duration_seconds < 2.5

    if is_short_signal:
        # GRID LAYOUT (e.g., 3x4 for 12 leads)
        cols = 4 if num_plots >= 4 else num_plots
        rows = math.ceil(num_plots / cols)

        # Compact size: ~4 inches width per col, ~3 inches height per row
        fig_width = 4 * cols
        fig_height = 3 * rows

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
        # Flatten axes array for easy iteration (handles 1D and 2D arrays)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]

    else:
        # VERTICAL STACK LAYOUT (1 column)
        cols = 1
        rows = num_plots

        # Wide size: at least 10 inches or 2 inches per second
        fig_width = max(12, duration_seconds * 2)
        fig_height = 2.0 * rows

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]

    # PLOTTING
    for i in range(len(axes_flat)):
        ax = axes_flat[i]

        # If we have more axes than leads (e.g. 3x4 grid for 10 leads), hide extra axes
        if i >= num_plots:
            ax.axis('off')
            continue

        lead_idx = leads[i]
        ax.plot(time, signal[:, lead_idx], linewidth=1.5 if is_short_signal else 1.0, color='#1f77b4')

        ax.set_title(f'{LEAD_NAMES[lead_idx]}', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)

        # Add labels only to outer plots in grid to reduce clutter
        if is_short_signal:
            if i % cols == 0:  # First column
                ax.set_ylabel('mV', fontsize=9)
            if i >= (rows - 1) * cols:  # Last row
                ax.set_xlabel('Time (s)', fontsize=9)
        else:
            ax.set_ylabel('mV', fontsize=9)
            if i == num_plots - 1:
                ax.set_xlabel('Time (s)', fontsize=10)

    plt.suptitle(f"{subtitle} ({duration_seconds:.2f}s)", fontsize=14, fontweight='bold',
                 y=0.98 if is_short_signal else 1.005)
    plt.tight_layout()
    plt.show()


def _format_record_label(record):
    base_label = record.label
    extra_labels = [str(label) for label in record.extra_labels if label]
    if extra_labels:
        return f"{base_label} ({', '.join(extra_labels)})"
    return str(base_label)


def draw_record(record, leads=None):
    if leads is None:
        leads = list(range(12))

    label_text = _format_record_label(record)
    subtitle = f"Record #{record.id}, Class={label_text}, Age={record.age:.2f}"

    num_samples = record.signal.shape[0]
    duration_seconds = num_samples / record.fs
    time = np.arange(num_samples) / record.fs

    # Calculate peak locations in seconds
    peak_times = record.r_peaks / record.fs

    num_plots = len(leads)

    # LAYOUT LOGIC
    is_short_signal = duration_seconds < 2.5

    if is_short_signal:
        # GRID LAYOUT
        cols = 4 if num_plots >= 4 else num_plots
        rows = math.ceil(num_plots / cols)
        fig_width = 4 * cols
        fig_height = 3 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]
    else:
        # VERTICAL STACK LAYOUT
        cols = 1
        rows = num_plots
        fig_width = max(12, duration_seconds * 2)
        fig_height = 2.0 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), sharex=True)
        axes_flat = axes.flatten() if num_plots > 1 else [axes]

    # PLOTTING
    for i in range(len(axes_flat)):
        ax = axes_flat[i]

        if i >= num_plots:
            ax.axis('off')
            continue

        lead_idx = leads[i]

        # Plot signal
        ax.plot(time, record.signal[:, lead_idx], linewidth=1.5 if is_short_signal else 1.0, color='#1f77b4')

        # Draw vertical dashed lines for R-peaks
        for p_time in peak_times:
            ax.axvline(x=p_time, color='red', linestyle='--', linewidth=1.0, alpha=0.7)

        ax.set_title(f'{LEAD_NAMES[lead_idx]}', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)

        if is_short_signal:
            if i % cols == 0:  # First column
                ax.set_ylabel('mV', fontsize=9)
            if i >= (rows - 1) * cols:  # Last row
                ax.set_xlabel('Time (s)', fontsize=9)
        else:
            ax.set_ylabel('mV', fontsize=9)
            if i == num_plots - 1:
                ax.set_xlabel('Time (s)', fontsize=10)

    # TITLE & LAYOUT ADJUSTMENT
    # 1. Add the title
    plt.suptitle(f"{subtitle} ({duration_seconds:.2f}s)", fontsize=14, fontweight='bold')

    # 2. Use rect to leave space at the top (top=0.96 leaves 4% space)
    # The rect format is [left, bottom, right, top] in normalized coordinates (0 to 1)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()
