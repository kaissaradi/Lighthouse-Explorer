
import numpy as np
from core.lh_qc_pipeline import QCResult, ValleyResult, SnippetResult, PCAKMeansResult, BLTRResult

def test_qc_result_properties():
    valley = ValleyResult(
        accepted=True,
        valley_low=-100.0,
        valley_high=-50.0,
        left_times=np.array([1, 2, 3]),
        left_vals=np.array([-110.0, -120.0, -115.0]),
        valley_times=np.array([10, 20]),
        valley_vals=np.array([-60.0, -70.0]),
        all_times=np.array([1, 2, 3, 10, 20]),
        all_vals=np.array([-110.0, -120.0, -115.0, -60.0, -70.0]),
        amp_hist_counts=np.array([1, 1, 1, 1, 1]),
        amp_hist_edges=np.array([-130, -110, -90, -70, -50, -30]),
        left_count=3,
        valley_count=2
    )
    
    snippets = SnippetResult(
        snippets=np.zeros((1, 60, 5)),
        times=np.array([1, 2, 3, 10, 20]),
        n_channels=1,
        snippet_len=60
    )
    
    pca_km = PCAKMeansResult(
        pca_coords=np.zeros((5, 3)),
        km_labels=np.array([0, 0, 0, 1, 1]),
        cluster_mean_waveforms=[np.zeros(60), np.zeros(60)],
        explained_variance_ratio=np.array([0.5, 0.3, 0.2]),
        n_pcs_used=3
    )
    
    bltr = BLTRResult(
        labels=np.array(['LH', 'LH', 'LH', 'soup', 'soup']),
        bl_bulk=np.array([0.8, 0.9, 0.85, 0.2, 0.1]),
        tr_bulk=np.array([0.1, 0.15, 0.1, 0.7, 0.8]),
        counts={'LH': 3, 'soup': 2},
        times=np.array([1, 2, 3, 10, 20])
    )
    
    qc_result = QCResult(
        channel=0,
        n_sorter_spikes=2,
        valley=valley,
        snippets=snippets,
        pca_km=pca_km,
        bltr=bltr
    )
    
    assert qc_result.n_total == 5
    assert qc_result.n_lh == 3
    assert qc_result.n_soup == 2
    assert qc_result.n_uncertain == 0
    # miss_rate = max(0, n_lh - n_sorter_spikes) / n_lh = (3 - 2) / 3 = 1/3
    assert abs(qc_result.miss_rate - 0.333333) < 1e-5
    assert qc_result.sorter_yield_ratio == 2/3
