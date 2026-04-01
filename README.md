# Predictive Coding — Inference Dynamics vs. Learning Rules in RSA

Extension of [Predictive Coding and the Visual Cortex](https://github.com/nilsleut/Predictive-Coding-and-the-Visual-Cortex) testing whether the cortical hierarchy gradient in RSA against 7T fMRI is produced by the **weight update rule** or by the **inference dynamics** of the PC network.

**Key finding: The gradient is determined by inference, not learning.** An untrained PC network with random weights produces the same hierarchy gradient as a fully trained one — suggesting the structure is inherited from the hierarchical ResNet initialization processed through PC's iterative inference phase.

---

## Background

The original PC network shows a crossing hierarchy gradient in RSA: r₀ correlates most strongly with V1 (ρ=0.30), r₃ with IT (ρ=0.16), with a significant Layer×ROI interaction (Δr₀−Δr₃ = +0.266, p=0.007). This project asks what produces that gradient:

1. **The weight update rule?** We test this by replacing standard Hebbian updates with Differential Hebbian (DH) updates — a biologically plausible STDP approximation proposed by Grewe et al. (2023).
2. **The inference dynamics?** We test this with a random baseline — a PC network with fixed random weights that only runs the inference phase, no weight updates at all.

---

## Results

### Interaction Effect Δr₀−Δr₃ across conditions (sub-01)

| Condition | Δr₀−Δr₃ | Retention |
|---|---|---|
| Standard PC (trained, Hebbian) | +0.269 | 100% (reference) |
| Differential Hebbian PC (trained) | +0.271 | 100.7% |
| **Random Baseline (untrained)** | **+0.271** | **100.6%** |

Results replicated across N=3 subjects (mean retention: Standard-PC +0.268, DH-PC +0.272, Random +0.271).

### RSA Profile — sub-01
![RSA Comparison](outputs/dh_sub-01/dh_01_rsa_comparison.png)

The grey dotted line ("Random, no training") overlaps almost perfectly with the DH-PC profile. Training improves absolute ρ values — Standard-PC r₀ at V1 is 0.309 vs. 0.206 for random — but the **crossing gradient structure is identical** across all three conditions.

### All three subjects

| Subject | Std-PC | DH-PC | Random |
|---|---|---|---|
| sub-01 | +0.269 | +0.271 | +0.271 |
| sub-02 | +0.216 | +0.218 | ~0.217 |
| sub-03 | +0.321 | +0.327 | ~0.326 |

---

## Interpretation

**The hierarchy gradient is not learned — it is inherited.**

The PC network is initialized from ResNet-50 features (layer1–layer4), which are themselves hierarchically organized. When the PC inference phase iteratively minimizes prediction errors across these four levels, it produces representations whose dissimilarity structure mirrors the cortical hierarchy — regardless of what the weights are.

Training (whether Hebbian or Differential Hebbian) improves absolute RSA correlations but does not change the hierarchical *structure* of the representations. The gradient emerges from the combination of:

1. **Hierarchical input structure** — ResNet layer1–layer4 features span low-to-high visual abstraction
2. **PC inference dynamics** — iterative top-down/bottom-up error minimization over T=30 steps

This is a negative result for the original hypothesis (DH rule preserves gradient) but a positive finding about PC inference: the multi-level prediction error minimization is sufficient to induce cortex-like representational geometry without any learning.

### What training does contribute

Training substantially improves absolute ρ values. At V1, Standard-PC r₀ achieves ρ=0.309 vs. ρ=0.206 for random — a +50% improvement. The learned weights help the network form better absolute representations of each cortical area, but the *relative ordering* across the hierarchy is already present in the random initialization.

### Convergence note

Standard-PC converges cleanly (Free Energy: ~1.7 → ~0.7). DH-PC does not converge under these hyperparameters — Early Stopping terminates after ~15 epochs with Free Energy ~3.0. This is an open limitation: Grewe et al. (2023) demonstrate DH convergence on synthetic FC networks; extending to high-dimensional hierarchical features likely requires a smaller `lr_w` or adaptive step-size.

![Convergence](outputs/dh_sub-01/dh_02_free_energy.png)

---

## Methods

**Architecture** — Hierarchical PC network (4 layers: 256→512→1024→2048) initialized from ResNet-50 features extracted from 720 THINGS images.

**Three conditions:**
- *Standard-PC*: Hebbian updates `dW ∝ ε @ r_higher.T`, trained to convergence
- *DH-PC*: Differential Hebbian `dW ∝ Δε @ r_higher.T` (Grewe 2023), T_infer_dh=5
- *Random Baseline*: Fixed random weights (seed=42), inference only, no updates

**RSA** — RDMs from r₀–r₃ activations for N=720 stimuli, compared to 7T fMRI RDMs (THINGS-fMRI, N=3 subjects, V1–IT) via Spearman ρ. Interaction effect Δr₀−Δr₃ = (mean ρ at V1/V2) − (mean ρ at LOC/IT) for r₀ minus same for r₃.

---

## Files

```
├── pc_differential_hebbian.py   — Main script (3 conditions)
├── predictve_coding_v8.py       — Base PC network (imported)
├── outputs/
│   ├── dh_sub-01/               — Plots + summary
│   ├── dh_sub-02/
│   └── dh_sub-03/
```

---

## Relation to Prior Work

Extends [Predictive Coding and the Visual Cortex](https://github.com/nilsleut/Predictive-Coding-and-the-Visual-Cortex) which established the hierarchy gradient under standard training. This project isolates the source of that gradient.

The finding that inference dynamics dominate over learning connects to broader questions in bio-plausible learning: if the representational geometry is largely determined by the inference structure, then the choice of weight update rule may matter less for alignment with neural data than previously assumed.

---

## References

Aceituno, P.V., Farinha, M.T., Loidl, R., & Grewe, B.F. (2023). Learning cortical hierarchies with temporal Hebbian updates. *Frontiers in Computational Neuroscience*, 17:1136010. [Paper](https://www.frontiersin.org/articles/10.3389/fncom.2023.1136010/full)

Aceituno, P.V., de Haan, S., Loidl, R., Beumer, L., & Grewe, B.F. (2025). Challenging backpropagation: Evidence for target learning in the neocortex. *bioRxiv*. [Preprint](https://www.biorxiv.org/content/10.1101/2024.04.10.588837)

Rao, R.P.N., & Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79–87.

---

## Portfolio

Part of a NeuroAI research portfolio built before university enrollment:
[github.com/nilsleut](https://github.com/nilsleut)
