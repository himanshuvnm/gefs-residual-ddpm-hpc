# Method Note: Conditional Residual DDPM for Atmospheric Ensembles

This workflow treats ensemble generation as a residual or perturbation modeling
problem. Given atmospheric context, the model generates ensemble-like
perturbation fields rather than only deterministic predictions.

The evaluation is ensemble-facing. Generated members are assessed not only by
field realism, but also by whether their spread and covariance-action behavior
are useful for atmospheric ensemble evaluation.

A key diagnostic is the matrix-free covariance action

```math
Bv = X(X^\top v)/(N_e-1),
```

where \(X\) contains ensemble perturbations. This allows localized covariance
responses to be evaluated without explicitly forming the dense covariance matrix.
