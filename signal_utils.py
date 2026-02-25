import numpy as np
from scipy import stats, signal

def calc_sqi_metrics(sig, fs):
    """
    Calculates Signal Quality Indices (SQIs) for all leads.
    """

    k_sqi = stats.kurtosis(sig, axis=0, fisher=True)
    s_sqi = stats.skew(sig, axis=0)

    # Fix for Welch warning: ensure nperseg <= signal length
    nperseg = min(1024, sig.shape[0])
    f, Pxx = signal.welch(sig, fs=fs, axis=0, nperseg=nperseg)

    qrs_band_mask = (f >= 5) & (f <= 15)
    broad_band_mask = (f >= 5) & (f <= 40)

    power_qrs = np.sum(Pxx[qrs_band_mask, :], axis=0)
    power_broad = np.sum(Pxx[broad_band_mask, :], axis=0)
    power_broad = np.maximum(power_broad, 1e-10)

    p_sqi = power_qrs / power_broad

    return {
        "kSQI": k_sqi,
        "sSQI": s_sqi,
        "pSQI": p_sqi
    }