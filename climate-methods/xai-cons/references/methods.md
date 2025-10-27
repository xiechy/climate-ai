# Observational Constraint Methods - Detailed Methodology

## Overview

This document provides comprehensive methodological details for Observational Constraint (Emergent Constraint, EC) analysis. It covers the theoretical foundation, mathematical formulations, and evaluation metrics used in XAI-Cons skill.

---

## Part I: Core Three-Step Framework

The observational constraint method typically relies on climate model multi-model ensembles (MME) and narrows future projection uncertainties by establishing statistical-physical linkages between observable historical quantities ($X$) and uncertain future projections ($Y$).

### 1. Establishing Emergent Relationships (High Inter-Model Correlation)

The first step in observational constraint is to identify and establish an **Emergent Relationship** - a significant statistical correlation between an observable historical climate variable (predictor $X$) and a future projection quantity (predictand $Y$) across the multi-model ensemble.

#### 1.1 Linear Relationship Assumption

Observational constraint methods typically assume a **linear relationship** to express the connection between $X$ and $Y$. For example, the relationship between future change $Y$ and historical observation $X$ can be expressed as:

$$Y = \bar{Y} + r(X - \bar{X})$$

Where:
- $Y$: Future projection variable
- $\bar{Y}$: Multi-model ensemble mean of $Y$
- $X$: Historical observable variable
- $\bar{X}$: Multi-model ensemble mean of $X$
- $r$: Inter-model correlation coefficient

#### 1.2 Importance of Correlation Coefficient

The magnitude of the inter-model correlation coefficient ($\rho$) directly determines the potential effectiveness of the constraint. If $\rho$ is small, the relative variance reduction (RRV) approaches zero. Therefore, **the correlation coefficient between $X$ and $Y$ must be statistically significant**, for example, at the $p < 0.05$ or $p < 0.10$ level.

**Key Requirements:**
- Statistical significance: typically $p < 0.05$
- Sufficient strength: generally $r > 0.3$ preferred
- Consistent across time periods (robustness)
- Physical plausibility of the relationship

---

### 2. Physical Mechanistic Understanding

Having a high inter-model correlation alone is insufficient. An emergent relationship must be supported by **reasonable physical mechanisms** to be recognized as a reliable **Emergent Constraint**.

#### 2.1 Inter-Model Regression (Revealing Physical Linkages and Key Drivers)

Inter-model regression analysis is a common approach to reveal physical linkages and identify key predictors ($X$).

##### 2.1.1 Identifying Uncertainty Patterns

First, conduct inter-model Empirical Orthogonal Function (EOF) analysis on the projection uncertainty of predictand $Y$ (e.g., East Asian summer precipitation change) to identify the primary **uncertainty modes** (e.g., PC1, PC2). This decomposes complex spatial field variations into a few physically meaningful modes.

**Mathematical Expression:**
$$Y(x,y,m) = \sum_{i=1}^{N} PC_i(m) \cdot EOF_i(x,y)$$

Where:
- $Y(x,y,m)$: Future projection for model $m$ at location $(x,y)$
- $EOF_i(x,y)$: $i$-th spatial pattern
- $PC_i(m)$: $i$-th principal component for model $m$

##### 2.1.2 Linking Historical to Future

Subsequently, regress the model-simulated **historical (current climate) mean state** or **trends** (i.e., potential predictor $X$) onto these uncertainty modes (PCi). Regression maps can reveal which current climate physical fields (e.g., SST, precipitation, circulation) simulated by models are most relevant to future projection uncertainty.

**Regression Analysis:**
$$PC_i = \beta \cdot X_{hist} + \epsilon$$

Where:
- $PC_i$: Uncertainty mode (principal component)
- $X_{hist}$: Historical climate variable
- $\beta$: Regression coefficient (reveals physical linkage strength)
- $\epsilon$: Residual

For example, regressing projected $\Delta Y$ onto historical $X$ can reveal **how model biases in simulating current climate states propagate into future climate projections**.

##### 2.1.3 Multi-Step Emergent Constraints

For complex systems, sometimes **multi-step emergent constraint methods** need to be developed. This involves first constraining **key physical processes** that affect the final predictand $Y$ (e.g., surface albedo feedback or different components of ENSO decay mechanisms), then using these constrained intermediate results to correct the final $Y$. This approach typically relies on **energy budget diagnostics** or **decomposition analysis** to clarify key physical drivers.

**Multi-Step Framework:**
$$X_{obs} \rightarrow P_1 \rightarrow P_2 \rightarrow ... \rightarrow Y_{constrained}$$

Where:
- $P_1, P_2, ...$: Intermediate physical processes
- Each step has its own emergent relationship and physical validation

---

### 3. Constraint Quality Assessment Indicators

After establishing and applying the observational constraint (i.e., substituting observed value $X_O$ into the emergent relationship to obtain constrained predictand $Y_C$), a series of indicators are needed to evaluate constraint quality and effectiveness.

#### 3.1 Narrowing Uncertainty Range (Reducing "Noise")

The core goal of observational constraint is to reduce projection uncertainty.

##### 3.1.1 Relative Variance Reduction (RRV/TRV)

This is the most direct indicator, measuring the percentage reduction in inter-model variance ($\sigma^2_Y$) after constraint. The constrained variance $\sigma^2_{Yc}$ formula includes original variance $\sigma^2_Y$, correlation coefficient $\rho$, and signal-to-noise ratio $SNR$:

$$\sigma^2_{Yc} = \left(1 - \frac{\rho^2}{1+SNR^{-1}}\right) \sigma^2_Y$$

Where:
- $\sigma^2_{Yc}$: Constrained variance
- $\sigma^2_Y$: Original (unconstrained) variance
- $\rho$: Correlation coefficient between $X$ and $Y$
- $SNR$: Signal-to-noise ratio (observational uncertainty vs. model spread)

**Total Reduced Variance (TRV)** is typically RRV multiplied by the percentage of variance explained (PCV) by that mode.

**Interpretation:**
- **Positive RRV** (0-100%): Constraint successfully reduces uncertainty
- **Negative RRV**: Constraint increases uncertainty (unreliable EC)
- **RRV > 30%**: Substantial uncertainty reduction
- **RRV < 10%**: Marginal improvement

#### 3.2 Constraint Result Reliability Assessment (Measuring "Skill")

To verify the accuracy and reliability of constraint results, typically use **Perfect Model Test** or **Leave-One-Out Cross-Validation Test**. In these tests, a single model is treated as a "pseudo-observation," then the remaining models are used to establish the constraint relationship and predict that pseudo-observed value.

##### 3.2.1 Root Mean Square Error (RMSE)

Measures the deviation between the constrained ensemble mean prediction and the "pseudo-observed value." **Lower RMSE** indicates more accurate constrained ensemble mean predictions. For example, in research constraining warming in China, constrained RMSE is generally lower than original projections.

$$RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(Y_{c,i} - Y_{obs,i})^2}$$

Where:
- $Y_{c,i}$: Constrained prediction for pseudo-observation $i$
- $Y_{obs,i}$: Actual value from pseudo-observation $i$
- $N$: Number of pseudo-observations (models)

##### 3.2.2 Continuous Ranked Probability Score (CRPS) and Skill Score (CRPSS)

CRPS evaluates the accuracy between the constrained probability distribution and pseudo-observed values. **Lower CRPS** indicates higher accuracy of probabilistic predictions.

$$CRPS = \int_{-\infty}^{\infty} [F(y) - H(y - y_{obs})]^2 dy$$

Where:
- $F(y)$: Cumulative distribution function of the constrained prediction
- $H(y - y_{obs})$: Heaviside function (step function at observation)
- $y_{obs}$: Observed value

CRPSS measures the skill score of constrained prediction relative to unconstrained prediction. **Positive CRPSS** (or higher skill score) indicates improved constraint prediction skill.

$$CRPSS = 1 - \frac{CRPS_{constrained}}{CRPS_{unconstrained}}$$

**Interpretation:**
- **CRPSS > 0**: Constraint improves skill
- **CRPSS = 0**: No improvement
- **CRPSS < 0**: Constraint degrades skill

##### 3.2.3 Spread/Error Ratio

Evaluates reliability of prediction ensemble. This ratio is the prediction ensemble standard deviation (Spread) divided by ensemble mean error (Error). **Ratio close to 1** indicates good prediction reliability.

$$Spread/Error = \frac{\sigma_{ensemble}}{\sqrt{\frac{1}{N}\sum_{i=1}^{N}(Y_{mean} - Y_{obs,i})^2}}$$

**Interpretation:**
- **< 1**: Overconfident (ensemble too narrow)
- **≈ 1**: Well-calibrated (ideal)
- **> 1**: Underconfident (ensemble too wide)

---

## Part II: Advanced Diagnostic Methods

### 4. Binning Analysis (Reliability Assessment)

Binning analysis evaluates EC relationship reliability by grouping relationships based on correlation coefficient strength.

#### 4.1 Method

1. **Collect EC relationships**: Gather many potential EC relationships (real or randomly generated)
2. **Bin by correlation**: Group relationships into bins based on correlation coefficient (e.g., r=0.3-0.4, 0.4-0.5, etc.)
3. **Calculate statistics**: For each bin, calculate:
   - Mean prior/posterior distribution
   - Variance reduction
   - Confidence intervals
4. **Evaluate your EC**: See which bin your EC falls into and its expected reliability

#### 4.2 Key Outputs

- **Prior/Posterior distribution comparison**: Shows how much constraint reduces uncertainty for given correlation strength
- **Credibility curves**: Relationship between correlation strength and constraint reliability
- **Your EC's position**: Where your r value falls on the reliability spectrum

**Reference Implementation:**
- `scripts/examples/src/binning_inference.py`

---

### 5. Residual Analysis (Isolating Unique Signals)

Residual analysis removes the linear influence of global mean signals to isolate region-specific teleconnections.

#### 5.1 Method

1. **Establish global relationship**:
   $$X_{regional} = \beta \cdot X_{global} + \epsilon$$

2. **Calculate residual**:
   $$X_{residual} = X_{regional} - (\beta \cdot X_{global} + intercept)$$

3. **Verify independence**:
   $$corr(X_{residual}, X_{global}) \approx 0$$

4. **Analyze residual teleconnections**:
   - Spatial correlation with global fields
   - Identify wave train pathways
   - Statistical significance testing

#### 5.2 Physical Interpretation

Three possible scenarios:

**Scenario A**: Strong residual correlation (r>0.3, p<0.05)
- Region acts as **dual-role hub**
- Both global indicator AND independent teleconnection source
- ~30-50% unique regional signal

**Scenario B**: Moderate residual correlation (0.1<r≤0.3)
- Primarily **global indicator** (~70-90%)
- Weak independent teleconnection (~10-30%)

**Scenario C**: Weak/no residual correlation (r≤0.1)
- Pure **global warming thermometer**
- No independent teleconnection mechanism
- Regional changes driven by global forcing

---

### 6. Lead-Lag Correlation Analysis

Examines temporal causality by calculating correlations at different time lags.

#### 6.1 Method

For time lags $\tau = -5, -4, ..., 0, ..., +4, +5$ years:

$$r(\tau) = corr(X(t), Y(t+\tau))$$

#### 6.2 Interpretation

- **Negative lag (τ<0)**: $X$ leads $Y$ (supports X→Y causality)
- **Zero lag (τ=0)**: Synchronous relationship (fast teleconnection or common driver)
- **Positive lag (τ>0)**: $Y$ leads $X$ (common forcing or bidirectional)

**Peak correlation timing reveals:**
- Response timescale of teleconnection
- Direction of influence
- Lag relationship between predictor and predictand

---

### 7. SVD Covariance Analysis

Identifies coupled spatial modes between predictor field and predictand field.

#### 7.1 Singular Value Decomposition

For predictor field $X(lat, lon, model)$ and predictand field $Y(lat, lon, model)$:

$$C = X^T Y$$

SVD: $C = U \Sigma V^T$

Where:
- $U$: Left singular vectors (predictor patterns)
- $V$: Right singular vectors (predictand patterns)
- $\Sigma$: Singular values (coupling strength)

#### 7.2 Purpose

- **Confirm large-scale coupling**: Not point-to-point artifacts
- **Identify dominant modes**: Most important covariance patterns
- **Spatial coherence**: Verify physical consistency of patterns

---

## Part III: Mathematical Formulations Summary

### Core EC Equation
$$Y_{future} = \beta \cdot X_{historical} + \epsilon$$

### Constrained Prediction
$$Y_{constrained} = \bar{Y} + \beta(X_{obs} - \bar{X})$$

### Variance Reduction
$$VR = 1 - \frac{\sigma^2_{posterior}}{\sigma^2_{prior}}$$

### Confidence Interval (66%)
$$CI_{66\%} = Y_{constrained} \pm 0.967 \cdot \sigma_{posterior}$$

### Confidence Interval (90%)
$$CI_{90\%} = Y_{constrained} \pm 1.645 \cdot \sigma_{posterior}$$

### Correlation Significance (t-test)
$$t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}}$$

Degrees of freedom: $df = n - 2$ (where $n$ = number of models)

---

## Part IV: Practical Guidelines

### When is an EC Relationship Reliable?

**Minimum Requirements:**
1. ✅ Statistical significance: $p < 0.05$
2. ✅ Sufficient correlation: $r > 0.3$ (preferably $r > 0.4$)
3. ✅ Positive variance reduction: $VR > 0$
4. ✅ Physical mechanism support
5. ✅ Robustness across tests (different periods, regions, datasets)

**Strong EC Characteristics:**
1. ⭐ $r > 0.6$, $p < 0.01$
2. ⭐ $VR > 30\%$
3. ⭐ Passes binning analysis (credible for that correlation strength)
4. ⭐ Better than random ECs
5. ⭐ Clear physical mechanism (not just statistical correlation)
6. ⭐ Consistent across perfect model tests (low RMSE, positive CRPSS)

### Common Pitfalls

**❌ Avoid:**
1. **Cherry-picking**: Testing many variable pairs, only reporting significant ones
2. **P-hacking**: Adjusting analysis until significance achieved
3. **Ignoring negative VR**: Must report and explain
4. **Weak physical justification**: Pure data mining without mechanism
5. **Extrapolation**: Applying constraint outside observed range
6. **Model dependence**: Few models driving entire correlation

---

## Part V: References and Further Reading

### Foundational Papers

**EC Methodology:**
1. Hall, A., Cox, P., Huntingford, C., & Klein, S. (2019). Progressing emergent constraints on future climate change. *Nature Climate Change*, 9, 269-278.
2. Klein, S. A., & Hall, A. (2015). Emergent constraints for cloud feedbacks. *Current Climate Change Reports*, 1(4), 276-287.

**EC Applications:**
1. Cox, P. M., et al. (2018). Emergent constraint on equilibrium climate sensitivity from global temperature variability. *Nature*, 553, 319-322.
2. Nijsse, F. J. M. M., et al. (2020). Emergent constraints on transient climate response (TCR) and equilibrium climate sensitivity (ECS) from historical warming in CMIP5 and CMIP6 models. *Earth System Dynamics*, 11, 737-750.

**Evaluation Methods:**
1. Brient, F. (2020). Evaluating the robustness of emergent constraints on climate sensitivity. *Journal of Climate*, 33(18), 7079-7099.
2. Caldwell, P. M., et al. (2018). Statistical significance of climate sensitivity predictors obtained by data mining. *Geophysical Research Letters*, 45, 1478-1488.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-27
**Maintainer:** Climate-AI Research Team
