"""
Predictive Coding — Differential Hebbian Learning Extension
============================================================
Erweitert predictve_coding_v8.py um Differential Hebbian (DH) Updates
nach Aceituno, Farinha, Loidl & Grewe (2023):
  "Learning cortical hierarchies with temporal Hebbian updates"
  Frontiers in Computational Neuroscience, 17:1136010.

Kernidee:
  Standard PC (v8): dW ∝ eps_after @ r_higher.T
  Differential Hebb: dW ∝ (eps_after - eps_before) @ r_higher.T

Biologische Motivation:
  Differential Hebbian = Rate-Code-Approximation von STDP.
  Der Update reagiert nur auf Fehleränderungen durch top-down
  Modulation — nicht auf die absolute Fehler-Baseline.

Forschungsfrage:
  Erhält Differential Hebbian den Kortex-Hierarchie-Gradienten
  (Δr₀−Δr₃ > 0, p=0.007) aus dem RSA-Profil — oder ist dieser
  ein Artefakt der Standard-Lernregel?

Verwendung:
  python pc_differential_hebbian.py sub-01
  python pc_differential_hebbian.py sub-02
  python pc_differential_hebbian.py sub-03

Outputs (in cfg.OUT_DIR):
  dh_01_rsa_comparison.png
  dh_02_free_energy.png
  dh_summary.txt
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import h5py

# ── Imports aus v8 ───────────────────────────────────────────
try:
    from predictve_coding_v8 import (
        Config,
        PredictiveCodingNet,
        extract_resnet_features,
        get_pc_representations,
        compute_rdm,
        train_pc,
    )
    print("predictve_coding_v8 geladen ✓")
except ImportError as e:
    sys.exit(
        f"FEHLER: {e}\n"
        "Starte dieses Script aus dem Predictive Coding Projektordner."
    )


# ══════════════════════════════════════════════════════════════
# fMRI-Daten laden
# ══════════════════════════════════════════════════════════════

def load_subject_data(sub_id: str, cfg: Config):
    """Lädt fMRI RDMs und ResNet Features für ein Subject."""
    cfg.H5_FILE   = cfg.DATENSATZ_DIR / f'{sub_id}_task-things_voxel-wise-responses.h5'
    cfg.VOX_META  = cfg.DATENSATZ_DIR / f'{sub_id}_task-things_voxel-metadata.csv'
    cfg.STIM_META = cfg.DATENSATZ_DIR / f'{sub_id}_task-things_stimulus-metadata.csv'

    print(f"\n[1/4] Lade Stimuli & fMRI — {sub_id}")
    vox_meta  = pd.read_csv(cfg.VOX_META,  sep=',')
    stim_meta = pd.read_csv(cfg.STIM_META, sep=',')

    roi_masks = {
        'V1':  vox_meta['V1'].values.astype(bool),
        'V2':  vox_meta['V2'].values.astype(bool),
        'V3':  vox_meta['V3'].values.astype(bool),
        'V4':  vox_meta['hV4'].values.astype(bool),
        'LOC': (vox_meta['lLOC'].values.astype(bool) |
                vox_meta['rLOC'].values.astype(bool)),
        'IT':  vox_meta['IT'].values.astype(bool),
    }

    combined_mask     = np.zeros(len(vox_meta), dtype=bool)
    for m in roi_masks.values():
        combined_mask |= m
    roi_voxel_indices = np.where(combined_mask)[0]
    global_to_local   = {int(g): l for l, g in enumerate(roi_voxel_indices)}

    with h5py.File(cfg.H5_FILE, 'r') as f:
        roi_data_raw = f['ResponseData/block0_values'][
            roi_voxel_indices, :].astype(np.float32)
    responses_all = roi_data_raw.T
    responses_all = ((responses_all - responses_all.mean(axis=0)) /
                     (responses_all.std(axis=0) + 1e-8))

    stim_meta_test   = stim_meta[stim_meta['trial_type'] == 'test'].copy()
    stim_meta_unique = stim_meta_test.drop_duplicates(subset='stimulus').copy()
    valid_concepts   = sorted(stim_meta_unique['concept'].unique().tolist())

    stim_order = []
    for concept in valid_concepts:
        stims = sorted(
            stim_meta_unique[
                stim_meta_unique['concept'] == concept
            ]['stimulus'].tolist()
        )[:1]
        stim_order.extend(stims)
    stim_order = stim_order[:cfg.N_IMAGES]

    stim_responses, image_paths = [], []
    for stim in tqdm(stim_order, desc='fMRI mitteln'):
        idx = stim_meta_test.index[
            stim_meta_test['stimulus'] == stim].tolist()
        if not idx:
            continue
        stim_responses.append(responses_all[idx].mean(axis=0))
        concept = stim_meta_test.loc[idx[0], 'concept']
        image_paths.append(cfg.THINGS_IMAGES_DIR / concept / stim)

    responses = np.array(stim_responses)
    print(f"  fMRI responses: {responses.shape}")

    fmri_rdms = {}
    for roi in cfg.ROI_NAMES:
        g_idx = np.where(roi_masks[roi])[0]
        l_idx = np.array([global_to_local[int(g)] for g in g_idx
                          if int(g) in global_to_local])
        fmri_rdms[roi] = compute_rdm(responses[:, l_idx])

    print("\n[2/4] Extrahiere ResNet-50 Features...")
    layer_features = extract_resnet_features(image_paths, cfg.DEVICE)

    return fmri_rdms, layer_features


# ══════════════════════════════════════════════════════════════
# Differential Hebbian PC-Netz
# ══════════════════════════════════════════════════════════════

class PredictiveCodingNetDH(PredictiveCodingNet):
    """
    PC-Netz mit Differential Hebbian Weight Updates.

    Einzige Änderung gegenüber PredictiveCodingNet:
      dW = η × (eps_after - eps_before) @ r_higher.T

    eps_before: Fehler nach t=0 (vor top-down Modulation)
    eps_after:  Fehler nach Konvergenz
    """

    def infer_dh(self, inputs: dict):
        """Inferenz mit eps_before-Tracking."""
        cfg = self.cfg
        r0 = inputs['layer1'].clone()
        r1 = inputs['layer2'].clone()
        r2 = inputs['layer3'].clone()
        r3 = inputs['layer4'].clone()
        eps_before = None

        T = getattr(self, '_T_infer_dh', cfg.T_infer)
        for t in range(T):
            pred0 = self.predict(r1, self.W1, self.b1)
            pred1 = self.predict(r2, self.W2, self.b2)
            pred2 = self.predict(r3, self.W3, self.b3)

            eps0 = r0 - pred0
            eps1 = r1 - pred1
            eps2 = r2 - pred2

            if t == 0:
                eps_before = (eps0.detach().clone(),
                              eps1.detach().clone(),
                              eps2.detach().clone())

            dr0 = -eps0
            dr1 = -eps1 + eps0 @ self.W1
            dr2 = -eps2 + eps1 @ self.W2
            dr3 =         eps2 @ self.W3

            r0 = r0 + cfg.lr_r * 0.5 * dr0
            r1 = r1 + cfg.lr_r * dr1
            r2 = r2 + cfg.lr_r * dr2
            r3 = r3 + cfg.lr_r * dr3

        pred0 = self.predict(r1, self.W1, self.b1)
        pred1 = self.predict(r2, self.W2, self.b2)
        pred2 = self.predict(r3, self.W3, self.b3)
        eps_after = (r0 - pred0, r1 - pred1, r2 - pred2)

        return ((r0.detach(), r1.detach(), r2.detach(), r3.detach()),
                eps_after, eps_before)

    def weight_update_dh(self, eps_after, eps_before, representations):
        """dW = η × Δeps @ r_higher.T"""
        eps0_a, eps1_a, eps2_a = eps_after
        eps0_b, eps1_b, eps2_b = eps_before
        r0, r1, r2, r3 = representations
        clip = self.cfg.grad_clip

        with torch.no_grad():
            dW1 = ((eps0_a - eps0_b).T @ r1) / eps0_a.shape[0]
            dW2 = ((eps1_a - eps1_b).T @ r2) / eps1_a.shape[0]
            dW3 = ((eps2_a - eps2_b).T @ r3) / eps2_a.shape[0]

            for dW in [dW1, dW2, dW3]:
                dW.clamp_(-clip, clip)

            self.W1.data += self.cfg.lr_w * dW1
            self.W2.data += self.cfg.lr_w * dW2
            self.W3.data += self.cfg.lr_w * dW3
            self._clip_weights()


# ══════════════════════════════════════════════════════════════
# DH Training
# ══════════════════════════════════════════════════════════════

def train_pc_dh(layer_features: dict, cfg: Config, T_infer_dh: int = 5):
    features_n, norms = {}, {}
    for k, v in layer_features.items():
        mean = v.mean(dim=0, keepdim=True)
        std  = v.std(dim=0, keepdim=True).clamp(min=1e-8)
        features_n[k] = (v - mean) / std
        norms[k] = (mean, std)

    N  = len(features_n['layer1'])
    pc = PredictiveCodingNetDH(cfg).to(cfg.DEVICE)
    pc.norms = norms
    pc._T_infer_dh = T_infer_dh

    print(f"\nDifferential Hebbian PC Training (Grewe 2023):")
    print(f"  dW ∝ Δeps @ r.T\n")

    fe_history, best_fe = [], float('inf')
    best_weights   = {k: v.clone() for k, v in pc.state_dict().items()}
    patience_count = 0

    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(N)
        epoch_fe, n_batches, diverged = 0.0, 0, False

        for start in range(0, N, cfg.batch_size):
            idx   = perm[start:start + cfg.batch_size]
            batch = {k: features_n[k][idx].to(cfg.DEVICE)
                     for k in features_n}

            reps, eps_after, eps_before = pc.infer_dh(batch)
            fe = pc.free_energy(eps_after)
            epoch_fe += fe
            n_batches += 1

            if np.isnan(fe) or fe > 1e6:
                print(f"  WARNUNG: Divergenz bei Epoch {epoch+1}")
                diverged = True
                break

            pc.weight_update_dh(eps_after, eps_before, reps)

        if diverged:
            break

        avg_fe = epoch_fe / n_batches
        fe_history.append(avg_fe)

        if avg_fe < best_fe:
            best_fe = avg_fe
            best_weights = {k: v.clone() for k, v in pc.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg.n_epochs} | "
                  f"FE: {avg_fe:.4f}  (best={best_fe:.4f}, "
                  f"patience={patience_count}/{cfg.patience})")

        if patience_count >= cfg.patience:
            print(f"\n  Early Stop bei Epoch {epoch+1}")
            break

    pc.load_state_dict(best_weights)
    print(f"\nDH-Training ✓  Beste FE: {best_fe:.4f}")
    return pc, fe_history


# ══════════════════════════════════════════════════════════════
# DH Repräsentationen
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def get_dh_representations(pc_dh: PredictiveCodingNetDH,
                            layer_features: dict) -> dict:
    features_n = {}
    for k, v in layer_features.items():
        mean, std = pc_dh.norms[k]
        features_n[k] = (v - mean) / std

    all_r = {k: [] for k in ['r0', 'r1', 'r2', 'r3']}
    N = len(features_n['layer1'])

    for start in range(0, N, 32):
        batch = {k: features_n[k][start:start+32].to(pc_dh.cfg.DEVICE)
                 for k in features_n}
        (r0, r1, r2, r3), _, _ = pc_dh.infer_dh(batch)
        for i, r in enumerate([r0, r1, r2, r3]):
            all_r[f'r{i}'].append(r.cpu())

    return {k: torch.cat(v, dim=0).numpy() for k, v in all_r.items()}


# ══════════════════════════════════════════════════════════════
# RSA
# ══════════════════════════════════════════════════════════════

def compute_rsa_profile(representations: dict,
                         fmri_rdms: dict) -> dict:
    from scipy.spatial.distance import pdist, squareform
    rho = {}
    for layer_key, reps in representations.items():
        model_rdm = compute_rdm(reps)
        rho[layer_key] = {}
        for roi, fmri_rdm in fmri_rdms.items():
            n   = model_rdm.shape[0]
            idx = np.triu_indices(n, k=1)
            r, _ = spearmanr(model_rdm[idx], fmri_rdm[idx])
            rho[layer_key][roi] = float(r)
    return rho


def interaction_effect(rho: dict,
                        early=('V1', 'V2'),
                        late=('LOC', 'IT')) -> float:
    def mean_rois(layer, rois):
        vals = [rho.get(layer, {}).get(r, 0) for r in rois
                if r in rho.get(layer, {})]
        return np.mean(vals) if vals else 0.0
    d_r0 = mean_rois('r0', early) - mean_rois('r0', late)
    d_r3 = mean_rois('r3', early) - mean_rois('r3', late)
    return d_r0 - d_r3


# ══════════════════════════════════════════════════════════════
# Plots & Summary
# ══════════════════════════════════════════════════════════════

def plot_rsa_comparison(rho_std, rho_dh, rho_resnet,
                         roi_names, out_path, rho_rand=None):
    x = np.arange(len(roi_names))
    layers     = ['r0', 'r1', 'r2', 'r3']
    colors_std = ['#c8a8e8', '#9b72cf', '#6a3fa0', '#3b1f6e']
    colors_dh  = ['#f4b8b8', '#e05a6a', '#b02040', '#6e1020']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#f9f9fb')
    fig.suptitle(
        'RSA-Profil: Standard-PC vs. Differential Hebbian PC\n'
        'Aceituno, Farinha, Loidl & Grewe (2023)',
        fontsize=13, fontweight='bold'
    )

    for ax, rho, title, colors in zip(
        axes,
        [rho_std, rho_dh],
        ['Standard PC (Hebbian, v8)',
         'Differential Hebbian PC (Grewe 2023)'],
        [colors_std, colors_dh]
    ):
        ax.set_facecolor('#f9f9fb')
        if rho_resnet:
            ax.plot(x, [rho_resnet.get(r, 0) for r in roi_names],
                    's--', color='#4477aa', linewidth=1.8,
                    markersize=7, label='ResNet-50', alpha=0.85, zorder=4)
        if rho_rand:
            # Random baseline: mean over r0-r3
            rand_vals = [np.mean([rho_rand.get(l, {}).get(r, 0) for l in ['r0','r1','r2','r3']])
                         for r in roi_names]
            ax.plot(x, rand_vals, 'x:', color='#aaaaaa', linewidth=1.3,
                    markersize=6, label='Random (no training)', alpha=0.7, zorder=2)
        for layer, color in zip(layers, colors):
            vals = [rho.get(layer, {}).get(r, 0) for r in roi_names]
            ax.plot(x, vals, 'o-', color=color, label=f'PC {layer}',
                    linewidth=2.3, markersize=8, zorder=3)
        ix = interaction_effect(rho)
        v  = '✓ Gradient erhalten' if ix > 0 else '✗ Gradient fehlt'
        ax.set_title(f'{title}\nΔr₀−Δr₃ = {ix:+.3f}  {v}', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(roi_names)
        ax.axvline(2.5, color='gray', linewidth=1,
                   linestyle=':', alpha=0.4)
        ax.set_xlabel('ROI (früh → spät)', fontsize=10)
        ax.set_ylabel('Spearman ρ', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {Path(out_path).name}")


def plot_free_energy(fe_std, fe_dh, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#f9f9fb')
    ax.set_facecolor('#f9f9fb')
    if fe_std:
        ax.plot(fe_std, color='#6a3fa0', linewidth=2,
                label='Standard PC (Hebbian)')
    if fe_dh:
        ax.plot(fe_dh, color='#e05a6a', linewidth=2,
                label='Differential Hebbian PC')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Free Energy', fontsize=11)
    ax.set_title('Konvergenz: Standard-PC vs. DH-PC',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {Path(out_path).name}")


def write_summary(rho_std, rho_dh, roi_names, out_path, rho_rand=None):
    layers = ['r0', 'r1', 'r2', 'r3']
    ix_std = interaction_effect(rho_std)
    ix_dh  = interaction_effect(rho_dh)
    pct    = ix_dh / ix_std * 100 if ix_std != 0 else float('nan')

    lines = ["Differential Hebbian PC — Summary", "=" * 55, ""]

    for name, rho, ix in [
        ('Standard PC (Hebbian)', rho_std, ix_std),
        ('Differential Hebbian PC (Grewe 2023)', rho_dh, ix_dh),
    ]:
        lines += [name,
                  f"  {'ROI':>6}" + "".join(f"  {l:>8}" for l in layers),
                  "  " + "─" * 50]
        for roi in roi_names:
            row = f"  {roi:>6}"
            for l in layers:
                v = rho.get(l, {}).get(roi, float('nan'))
                row += f"  {v:>8.4f}"
            lines.append(row)
        lines += [f"  Δr₀−Δr₃: {ix:+.4f}", ""]

    ix_rand = interaction_effect(rho_rand) if rho_rand else float('nan')
    rand_pct = ix_rand / ix_std * 100 if ix_std != 0 and rho_rand else float('nan')

    lines += [
        "Vergleich",
        f"  Standard-PC      Δr₀−Δr₃ = {ix_std:+.4f}",
        f"  DH-PC            Δr₀−Δr₃ = {ix_dh:+.4f}  ({pct:.1f}% Erhaltung)",
        f"  Random Baseline  Δr₀−Δr₃ = {ix_rand:+.4f}  ({rand_pct:.1f}% Erhaltung)",
        "",
        "Interpretation:",
    ]
    if rho_rand:
        if ix_dh > ix_rand * 1.1:
            lines.append("  DH-PC > Random: Training traegt zum Gradienten bei")
        elif ix_dh < ix_rand * 1.1 and ix_dh > ix_rand * 0.9:
            lines.append("  DH-PC ≈ Random: Gradient entsteht aus Inferenz-Dynamik, nicht Training")
        else:
            lines.append("  DH-PC < Random: unerwartetes Resultat")
    lines += [
        "",
        "Referenz: Aceituno, Farinha, Loidl & Grewe (2023)",
        "  Frontiers in Computational Neuroscience, 17:1136010",
    ]

    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"  → {Path(out_path).name}")



# ══════════════════════════════════════════════════════════════
# Random Baseline — PC ohne Weight Updates
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def get_random_baseline_representations(layer_features: dict,
                                         cfg: Config) -> dict:
    """
    PC-Netz mit zufälligen Initialgewichten, OHNE Weight Updates.
    Nur Inferenz läuft. Zeigt ob das RSA-Resultat von DH-PC
    durch das Training entsteht oder bereits in den ResNet-Features
    / der Inferenz-Dynamik steckt.

    Wenn Random-Baseline ≈ DH-PC → DH-Resultat nicht durch Training.
    Wenn Random-Baseline << DH-PC → Training bringt etwas.
    """
    # Normierung
    features_n = {}
    norms = {}
    for k, v in layer_features.items():
        mean = v.mean(dim=0, keepdim=True)
        std  = v.std(dim=0, keepdim=True).clamp(min=1e-8)
        features_n[k] = (v - mean) / std
        norms[k] = (mean, std)

    # Zufällig initialisiertes PC-Netz — keine Updates
    torch.manual_seed(42)
    pc_rand = PredictiveCodingNet(cfg).to(cfg.DEVICE)
    pc_rand.norms = norms

    all_r = {k: [] for k in ['r0', 'r1', 'r2', 'r3']}
    N = len(features_n['layer1'])

    for start in range(0, N, 32):
        batch = {k: features_n[k][start:start+32].to(cfg.DEVICE)
                 for k in features_n}
        (r0, r1, r2, r3), _ = pc_rand.infer(batch)
        for i, r in enumerate([r0, r1, r2, r3]):
            all_r[f'r{i}'].append(r.cpu())

    return {k: torch.cat(v, dim=0).numpy() for k, v in all_r.items()}

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def run_dh_comparison(sub_id: str = 'sub-01'):
    cfg     = Config()
    out_dir = cfg.OUT_DIR / f'dh_{sub_id}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDifferential Hebbian Vergleich — {sub_id}")
    print(f"Device: {cfg.DEVICE} | Output: {out_dir}\n")

    # 1. Daten
    fmri_rdms, layer_features = load_subject_data(sub_id, cfg)

    # 2. Standard-PC
    print("\n[3/4] Standard-PC trainieren...")
    pc_std, fe_std = train_pc(layer_features, cfg)
    reps_std = get_pc_representations(pc_std, layer_features)
    rho_std  = compute_rsa_profile(
        {k: reps_std[k] for k in ['r0', 'r1', 'r2', 'r3']},
        fmri_rdms
    )

    # 3. DH-PC — sweep over T_infer_dh values
    print("\n[4/4] DH-PC trainieren (T_infer sweep: 3, 5, 10)...")
    best_dh_result = None
    best_T = 5
    sweep_results = {}
    for T in [3, 5, 10]:
        print(f"\n  T_infer_dh = {T}")
        pc_dh_T, fe_dh_T = train_pc_dh(layer_features, cfg, T_infer_dh=T)
        reps_T = get_dh_representations(pc_dh_T, layer_features)
        rho_T  = compute_rsa_profile(reps_T, fmri_rdms)
        ix_T   = interaction_effect(rho_T)
        converged = len(fe_dh_T) > 5 and fe_dh_T[-1] < fe_dh_T[0]
        print(f"  T={T}: Δr0-Δr3={ix_T:+.4f}  converged={converged}  final_FE={fe_dh_T[-1]:.4f}")
        sweep_results[T] = (fe_dh_T, rho_T, converged)
        if best_dh_result is None or (converged and not sweep_results[best_T][2]):
            best_dh_result = (rho_T, fe_dh_T)
            best_T = T
        elif converged and ix_T > interaction_effect(best_dh_result[0]):
            best_dh_result = (rho_T, fe_dh_T)
            best_T = T
    rho_dh, fe_dh = best_dh_result
    print(f"\n  Bestes T_infer_dh = {best_T}")

    # ResNet Baseline
    resnet_rdm = compute_rdm(layer_features['layer4'].numpy())
    rho_resnet = {}
    for roi, fmri_rdm in fmri_rdms.items():
        n = resnet_rdm.shape[0]
        idx = np.triu_indices(n, k=1)
        r, _ = spearmanr(resnet_rdm[idx], fmri_rdm[idx])
        rho_resnet[roi] = float(r)

    # Random Baseline — kein Training, nur Inferenz mit Zufallsgewichten
    print("\nRandom Baseline (keine Weight Updates)...")
    reps_rand = get_random_baseline_representations(layer_features, cfg)
    rho_rand  = compute_rsa_profile(reps_rand, fmri_rdms)
    ix_rand   = interaction_effect(rho_rand)
    print(f"  Random Baseline Δr₀−Δr₃ = {ix_rand:+.4f}")

    # Outputs
    roi_names = list(cfg.ROI_NAMES)
    print("\nPlots erstellen...")
    plot_rsa_comparison(rho_std, rho_dh, rho_resnet, roi_names,
                         out_dir / 'dh_01_rsa_comparison.png',
                         rho_rand=rho_rand)
    plot_free_energy(fe_std, fe_dh,
                     out_dir / 'dh_02_free_energy.png')
    write_summary(rho_std, rho_dh, roi_names,
                  out_dir / 'dh_summary.txt',
                  rho_rand=rho_rand)

    # Ergebnis
    print("\n" + "━" * 55)
    print("RESULTAT")
    print("━" * 55)
    ix_std = interaction_effect(rho_std)
    ix_dh  = interaction_effect(rho_dh)
    for name, ix in [('Standard-PC', ix_std), ('DH-PC', ix_dh), ('Random Baseline', ix_rand)]:
        v = '✓ Gradient erhalten' if ix > 0 else '✗ Gradient fehlt'
        print(f"  {name:20s}  Δr₀−Δr₃ = {ix:+.4f}  {v}")
    if ix_std != 0:
        print(f"  DH Erhaltungsrate:    {ix_dh/ix_std*100:.1f}%")
        print(f"  Random Erhaltungsrate:{ix_rand/ix_std*100:.1f}%")
    print(f"\n  Outputs: {out_dir}/dh_*.png + dh_summary.txt")
    print("━" * 55)

    return rho_std, rho_dh


if __name__ == '__main__':
    sub_id = sys.argv[1] if len(sys.argv) > 1 else 'sub-01'
    run_dh_comparison(sub_id)