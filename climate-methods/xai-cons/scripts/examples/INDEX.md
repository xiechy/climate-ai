# XAI in Climate Science - Exemplar Papers

## Purpose
This index catalogs high-quality papers demonstrating best practices for applying XAI to climate problems. Each entry includes methodology highlights and key takeaways.

---

## 📊 By Application Domain

### 🌡️ Temperature Extremes
#### 1. Neural Network Attribution of Heatwaves
- **Paper:** Barnes et al. (2020), "Viewing Forced Climate Patterns Through an AI Lens"
- **Journal:** Geophysical Research Letters
- **PDF:** `papers/barnes_2020_cam_climate.pdf`
- **XAI Method:** Class Activation Mapping (CAM)
- **Climate Problem:** Forced response pattern identification
- **Key Innovation:** Automated detection of climate change "fingerprints"

**Methodology Highlights:**
- Trained CNN on climate model output (forced + control)
- Used CAM to visualize regions driving classification
- Validated against known physics (Arctic amplification pattern)
- Tested on multiple models for robustness

**Takeaways for Your Research:**
1. CAM works well for spatial pattern detection in gridded climate data
2. Always validate XAI patterns against established physical understanding
3. Test on held-out models to ensure generalization
4. Compare forced pattern to internal variability

---

### ☁️ Clouds and Radiation
#### 2. SHAP Analysis of Cloud Feedbacks
- **Paper:** Toms et al. (2020), "Physically Interpretable Neural Networks..."
- **Journal:** Journal of Advances in Modeling Earth Systems
- **PDF:** `papers/toms_2020_neural_net_clouds.pdf`
- **XAI Method:** SHAP + Physical constraints
- **Climate Problem:** Cloud parameterization

**Methodology Highlights:**
- Embedded physical constraints in NN architecture
- Used SHAP to quantify variable importance
- Compared SHAP values to sensitivity tests from traditional model

**Takeaways:**
1. SHAP values align well with traditional sensitivity analysis
2. Embedding physics reduces need for post-hoc XAI
3. Always compare XAI to benchmark methods
...

---

### 🔥 Extreme Events
#### 3. LRP for Extreme Precipitation
- **Paper:** Ebert-Uphoff et al. (2021)
- **Journal:** Environmental Data Science
- **XAI Method:** Layer-wise Relevance Propagation
- **Climate Problem:** Precipitation nowcasting

**Relevant for Your Research:**
- Observation-constrained training (radar data, not model output)
- Physical consistency checks (moisture sources, orographic effects)
- Spatial patterns more interpretable than global importance

---

## 🔍 By XAI Method

### SHAP
- ✅ Toms et al. (2020) - Cloud feedbacks
- ✅ Sonnewald et al. (2021) - Ocean eddies

### LRP
- ✅ Ebert-Uphoff et al. (2021) - Precipitation
- ✅ Mayer et al. (2022) - Storm tracking

### CAM/Grad-CAM
- ✅ Barnes et al. (2020) - Forced patterns
- ✅ Labe & Barnes (2021) - Arctic amplification

---

## 🎯 By Methodological Focus

### Physical Validation
- Barnes et al. (2020) - Arctic amplification matches theory
- Toms et al. (2020) - Comparison to sensitivity tests
- **⭐ Your wildfire paper (2022)** - Observation constraints

### Multi-Model Analysis
- **⭐ Your wildfire paper** - 13 CMIP6 models
- Barnes et al. (2020) - CMIP5 ensemble

### Uncertainty Quantification
- Sonnewald et al. (2021) - Bootstrap SHAP values
- **⭐ Your Arctic paper (2024)** - Internal variability separation

---

## Quick Reference

| What Are You Doing? | Consult This Paper |
|---------------------|-------------------|
| Applying CAM to spatial patterns | ✅ Barnes+ 2020 |
| Using SHAP for feature importance | ✅ Toms+ 2020 |
| LRP on CNNs for climate | ✅ Ebert-Uphoff+ 2021 |
| Validating XAI with physics | ✅ All above |
| Observation-constrained ML | ✅ Your wildfire 2022 |
| Separating forced vs. internal variability | ✅ Barnes+ 2020, Your Arctic 2024 |