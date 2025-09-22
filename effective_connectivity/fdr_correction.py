import numpy as np
from scipy.stats import false_discovery_control
from statsmodels.stats.multitest import multipletests
import pandas as pd

# P-values from regression analysis (excluding overall model statistics)
p_values = [
    0.388,  # Intercept
    0.394,  # eyes open
    0.004,  # CEN→DMN **
    0.137,  # CEN→SN
    0.459,  # DMN→CEN
    0.308,  # DMN→DMN
    0.504,  # DMN→SN
    0.455,  # SN→CEN
    0.792,  # SN→DMN
    0.831,  # SN→SN
    0.597,  # beta
    0.404,  # delta
    0.414,  # gamma
    0.466,  # theta
    0.919,  # CEN→DMN × beta
    0.350,  # CEN→SN × beta
    0.308,  # DMN→CEN × beta
    0.428,  # DMN→DMN × beta
    0.523,  # DMN→SN × beta
    0.254,  # SN→CEN × beta
    0.818,  # SN→DMN × beta
    0.319,  # SN→SN × beta
    0.156,  # CEN→DMN × delta
    0.664,  # CEN→SN × delta
    0.255,  # DMN→CEN × delta
    0.588,  # DMN→DMN × delta
    0.765,  # DMN→SN × delta
    0.424,  # SN→CEN × delta
    0.377,  # SN→DMN × delta
    0.711,  # SN→SN × delta
    0.491,  # CEN→DMN × gamma
    0.976,  # CEN→SN × gamma
    0.008,  # DMN→CEN × gamma **
    0.492,  # DMN→DMN × gamma
    0.650,  # DMN→SN × gamma
    0.472,  # SN→CEN × gamma
    0.539,  # SN→DMN × gamma
    0.494,  # SN→SN × gamma
    0.426,  # CEN→DMN × theta
    0.354,  # CEN→SN × theta
    0.172,  # DMN→CEN × theta
    0.188,  # DMN→DMN × theta
    0.942,  # DMN→SN × theta
    0.305,  # SN→CEN × theta
    0.862,  # SN→DMN × theta
    0.099   # SN→SN × theta
]

# Variable names
variable_names = [
    'Intercept', 'eyes_open', 'CEN→DMN', 'CEN→SN', 'DMN→CEN', 'DMN→DMN', 
    'DMN→SN', 'SN→CEN', 'SN→DMN', 'SN→SN', 'beta', 'delta', 'gamma', 'theta',
    'CEN→DMN×beta', 'CEN→SN×beta', 'DMN→CEN×beta', 'DMN→DMN×beta', 'DMN→SN×beta',
    'SN→CEN×beta', 'SN→DMN×beta', 'SN→SN×beta', 'CEN→DMN×delta', 'CEN→SN×delta',
    'DMN→CEN×delta', 'DMN→DMN×delta', 'DMN→SN×delta', 'SN→CEN×delta', 
    'SN→DMN×delta', 'SN→SN×delta', 'CEN→DMN×gamma', 'CEN→SN×gamma', 
    'DMN→CEN×gamma', 'DMN→DMN×gamma', 'DMN→SN×gamma', 'SN→CEN×gamma',
    'SN→DMN×gamma', 'SN→SN×gamma', 'CEN→DMN×theta', 'CEN→SN×theta',
    'DMN→CEN×theta', 'DMN→DMN×theta', 'DMN→SN×theta', 'SN→CEN×theta',
    'SN→DMN×theta', 'SN→SN×theta'
]

# Method 1: Using scipy (Benjamini-Hochberg)
fdr_corrected_scipy = false_discovery_control(p_values, method='bh')

# Method 2: Using statsmodels (more options)
rejected, fdr_corrected_sm, alpha_sidak, alpha_bonf = multipletests(
    p_values, alpha=0.05, method='fdr_bh'
)

# Create results dataframe
results = pd.DataFrame({
    'Variable': variable_names,
    'P_value': p_values,
    'FDR_corrected': fdr_corrected_sm,
    'Significant_FDR': rejected
})

# Display results
print("FDR Correction Results (α = 0.05):")
print("=" * 50)
print(results.to_string(index=False))

# Show only significant results after FDR correction
significant = results[results['Significant_FDR']]
print(f"\nSignificant after FDR correction ({len(significant)} out of {len(p_values)}):")
print("=" * 60)
print(significant.to_string(index=False))

# Alternative methods
print(f"\nOther correction methods:")
print(f"Bonferroni: {alpha_bonf:.6f}")
print(f"Šidák: {alpha_sidak:.6f}")