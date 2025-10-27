# Exemplar Peer Review Catalog

## Overview

This directory contains high-quality peer review examples from Nature journals, focusing on climate science, AI/ML applications, and methodological rigor. These examples serve as benchmarks for comprehensive, constructive peer review.

## Purpose

Use these exemplar reviews to:
- Learn effective review structure and tone
- Understand depth of critique expected for high-impact journals
- Identify best practices for technical evaluation
- Develop templates for specific subfields
- Calibrate expectations for rigor and completeness

---

## Catalog of Exemplar Reviews

### 🔥 Climate Attribution Studies

#### [To be added]
- **Journal:** 
- **Year:** 
- **Topic:** Extreme event attribution / detection and attribution
- **Key Strengths of Review:**
  - [Awaiting first attribution study review]
- **Notable Sections:**
  - 
- **Relevant for:** 
- **Key Takeaways:**
  - 

---

### 🤖 Deep Learning / AI in Climate Science

#### 1. Machine Learning-Based Observation-Constrained Wildfire Projections
- **Journal:** Nature Communications
- **Year:** 2022
- **Paper Title:** "Machine learning-based observation-constrained projections reveal elevated global socioeconomic risks from wildfire"
- **PDF Filename:** `nat_commun_2022_ml_wildfire_projections.pdf`
- **Topic:** Machine learning + emergent constraints for global wildfire carbon emission projections

**Key Strengths of Review:**
- In-depth assessment of emergent constraints (EC) method applicability and limitations
- Detailed discussion of machine learning extrapolation issues and data space coverage verification
- Emphasis on observational data quality impact on constraint results (tiered: temperature/precipitation > humidity/soil > wind speed)
- Clear requirement to report uncertainty sources (model independence, observational errors, tipping points, feedbacks)
- Constructive demands for terrain variables and multi-scenario analysis (SSP2-4.5 vs SSP5-8.5)
- Three reviewers providing complementary feedback from different perspectives (methodology, EC theory, fire physics)

**Notable Sections:**

**Methodological Rigor (Reviewer 1):**
- Detailed ML parameter optimization requirements:
  - Random Forest: number of trees, variables per split (mtry)
  - SVM: kernel type, cost parameter (C), kernel parameter (sigma)
  - GBM: tree complexity, learning rate, number of iterations
- Questioned comparability of variable importance metrics across different ML algorithms
- Suggested jacknife estimators for unified variable importance assessment
- Concerned about NorESM models' (sharing CLM5 land component) influence on overall trends

**Emergent Constraints Theory (Reviewer 2):**
- Systematic discussion of traditional EC pitfalls (citing Hall et al. 2019, Williamson et al. 2021, Sanderson et al. 2021):
  - Insufficient statistical relationship strength (large fire-prone areas without significant correlation)
  - Sample size insufficiency
  - Model non-independence (multiple ESMs sharing components)
  - Overlooking tipping points (Lenton et al. 2008)
  - Unclear mechanisms (ML methods obscure physical processes)
- Emphasized observational data quality differences: wind speed, relative humidity, lightning less reliable than temperature/precipitation
- Warned against "strong but overconfident" constraints risk (Sanderson et al. 2021)
- Required extrapolation uncertainty visualization: demonstrate observational data space coverage by training data

**Wildfire Physical Processes (Reviewer 3):**
- **Terrain factors**: Highlighted terrain ruggedness more important than raw elevation
  - Definition: local relief (max-min elevation difference within ~1.5km radius)
  - References: Di Virgilio et al. 2019, Sharples et al. 2016, Abram et al. 2021
  - Australian 2019-20 fires case: terrain ruggedness + forest fuels + critically low fuel moisture
- **Fuel representation limitations**: Satellite LAI cannot capture surface and near-surface forest fuels (most critical for fire)
- **Threshold behavior**: Fire behavior changes abruptly below certain fuel moisture thresholds (spotting), ML may miss this
- **Carbon feedback missing**: Fire carbon emissions constitute important climate feedback, offline prediction not coupled to climate system

**Spatial Validation Requirements:**
- Not just global R², but identify regions with weaker performance
- Display spatial distribution of model bias (bias maps), not just model vs. observation values
- Distinguish fire types (forest fire, grass fire, peat fire) for separate assessment

**Scenario Dependency Testing:**
- Required testing conservative scenario (SSP2-4.5) to verify method sensitivity to different emission pathways
- Physical interpretation of differences between SSP2-4.5 and SSP5-8.5 results
- Acknowledge SSP5-8.5 is not "business as usual" but a high-emission scenario

**Highlights of Author Response:**
- Added terrain variable (orography) but results changed little → suggests historical fire emissions implicitly contain terrain information
- New Extended Data Fig. 14 showing observational vs. simulated data space coverage → proves minimal extrapolation error
- Sensitivity test excluding NorESM models → quantifies model dependency
- 10-fold cross-validation R² > 0.8 → demonstrates in-sample fitting quality
- Explicitly discussed limitations rather than avoiding them

**Relevant for:** 
- Machine learning + climate prediction research
- Emergent constraints method applications
- Observational data constraining ESMs
- Wildfire/extreme event attribution studies
- Uncertainty quantification
- Multi-model ensemble analysis

**Key Takeaways:**

**Methodological Essentials:**
1. **Extrapolation check indispensable**: Must demonstrate observational data space sufficiently covered by training data (scatterplots comparing all predictor variables)
2. **ML parameter full transparency**: Report all hyperparameters, optimization methods, cross-validation strategies—cannot just say "we used Random Forest"
3. **Multi-scenario testing**: Test at least two emission scenarios to verify method robustness
4. **Unified variable importance method**: Different ML algorithms' importance metrics not directly comparable; consider jacknife or permutation methods

**Uncertainty Sources Must Be Discussed:**
5. **Observational data quality tiers**: Clarify which observational variables are reliable (temperature/precipitation) vs. highly uncertain (wind/humidity)
6. **Model independence**: Identify ESMs sharing components (e.g., CLM5), acknowledge reduced effective sample size
7. **Sub-grid physical processes**: Slope, aspect, terrain ruggedness (~1.5km) important for extreme events but difficult for coarse-resolution ESMs to capture
8. **Tipping points/threshold behavior**: ML struggles with discontinuous physical processes (e.g., fuel moisture thresholds, ecosystem tipping points)

**Validation Strategies:**
9. **Spatial performance stratified reporting**: Not just global metrics, show which regions perform well vs. poorly
10. **Historical validation period**: Using observations to constrain near-future (2007-2016) proves method effectiveness, but long-term prediction relies on ESM physical process accuracy
11. **Fire type classification**: Forest, grassland, peat fires have different drivers; separate validation more meaningful

**Integration with Physical Processes:**
12. **Mechanistic interpretability**: ML provides predictions but obscures mechanisms; need to combine variable importance analysis with known physical processes for validation
13. **Feedback coupling future direction**: Offline prediction is first step; future needs to couple observation-constrained results back into climate-ecosystem models

**Paper Writing Insights:**
14. **Honest discussion of limitations**: Authors greatly expanded limitations section in response (from few sentences to full paragraph), winning reviewer approval
15. **Constructive response strategy**: When unable to fully meet requirements (e.g., sub-grid terrain), acknowledge limitations and point to future directions

---

### 📊 Climate Projections and Uncertainty Quantification

#### [To be added]
- **Journal:** 
- **Year:** 
- **Topic:** Multi-model ensemble projections, emergent constraints
- **Key Strengths of Review:**
  - [Awaiting first projection study review]
- **Notable Sections:**
  - 
- **Relevant for:** 
- **Key Takeaways:**
  - 

---

### 🔍 Observational Data Analysis

#### [To be added]
- **Journal:** 
- **Year:** 
- **Topic:** Satellite/reanalysis data analysis, trend detection
- **Key Strengths of Review:**
  - [Awaiting first observational data review]
- **Notable Sections:**
  - 
- **Relevant for:** 
- **Key Takeaways:**
  - 

---

### 🧠 Explainable AI (XAI) in Earth System Science

#### [To be added]
- **Journal:** 
- **Year:** 
- **Topic:** Interpretable ML, feature importance, mechanistic insights
- **Key Strengths of Review:**
  - [Awaiting first XAI review]
- **Notable Sections:**
  - 
- **Relevant for:** 
- **Key Takeaways:**
  - 

---

## Common Standards Across Exemplar Reviews

### Methodological Rigor Requirements
1. **Reproducibility**
   - Code and data availability expectations
   - Computational environment documentation
   - Random seed reporting for stochastic methods

2. **Statistical Rigor**
   - Multiple testing correction
   - Effect size reporting with uncertainty
   - Appropriate baseline comparisons
   - Out-of-sample validation

3. **Physical Consistency**
   - Energy/mass conservation checks
   - Consistency with established climate dynamics
   - Physical plausibility of results
   - Mechanism evaluation beyond correlation

4. **Uncertainty Quantification**
   - Multiple sources of uncertainty considered
   - Appropriate propagation through analysis chain
   - Sensitivity analyses to key assumptions
   - Confidence intervals or credible intervals reported

### Reporting Standards for Climate-ML Papers
- Model architecture details (layer dimensions, activation functions)
- Hyperparameter search strategy
- Training details (optimizer, learning rate schedule, batch size)
- Data preprocessing and normalization
- Train/validation/test split rationale
- Computational resources and training time
- Performance metrics on held-out test set
- Comparison to relevant baselines (physics-based and ML)
- Ablation studies to justify design choices

### Key Questions Consistently Raised

**For Attribution Studies:**
- How are observational uncertainties propagated?
- Is the counterfactual definition appropriate?
- Are models validated for the specific variable/region?
- How is model independence assessed?

**For ML/Climate Studies:**
- Does the model learn physically meaningful features?
- How does performance degrade with distribution shift?
- Are predictions physically consistent (e.g., thermodynamic constraints)?
- How does the method compare to domain-adapted baselines?

**For Projection Studies:**
- How is model performance/independence assessed?
- Are emergent constraints validated out-of-sample?
- How sensitive are results to model selection?
- Are multiple lines of evidence synthesized?

---

## Usage Guidelines

### For Conducting Reviews

1. **Select Relevant Exemplar:** Choose review(s) similar to manuscript topic
2. **Identify Key Evaluation Criteria:** Note specific technical requirements
3. **Calibrate Depth:** Match level of critique to journal tier and manuscript scope
4. **Adapt Language:** Maintain constructive tone while being rigorous

### For Self-Review Before Submission

1. **Pre-submission Check:** Compare your manuscript against standards in relevant exemplars
2. **Anticipate Concerns:** Address common issues raised in similar reviews
3. **Strengthen Methods:** Ensure key details highlighted in exemplars are included
4. **Validate Thoroughly:** Ensure validation approaches match field expectations

---

## Maintaining This Catalog

### When Adding New Exemplar Reviews:

1. **Add PDF:** Place file in this directory with descriptive filename
   - Format: `journal_year_topic_keywords.pdf`
   - Example: `nature_climate_2024_attribution_heatwaves.pdf`

2. **Update This INDEX:** Add entry using template above
   - Fill in all metadata fields
   - Identify 3-5 key strengths
   - Extract 2-3 actionable takeaways
   - Note relevant applications

3. **Extract Reusable Templates:** If review structure is particularly effective, create a template in `../review_templates/`

4. **Cross-Reference:** Link to relevant sections in `SKILL.md`, `common_issues.md`, or `reporting_standards.md`

---

## Maintenance Notes

- **Last Updated:** October 2025
- **Total Reviews:** 1
- **Coverage Gaps:** Areas needing more examples
  - Paleoclimate modeling
  - Climate economics/IAMs
  - Regional climate modeling
  - Coupled model development
  - High-resolution process studies
  - Extreme event attribution (needed)
  - XAI method validation (needed)
  - Observational constraint methods (have 1)

---

## Related Resources

- See `../SKILL.md` for systematic peer review workflow
- See `../common_issues.md` for frequent methodological problems
- See `../reporting_standards.md` for discipline-specific guidelines (PRISMA, CONSORT, etc.)
- See `../review_templates/` for structured review templates by subdiscipline

---

## Quick Reference: When Is This Exemplar Useful?

| What Are You Doing? | Consult This Review |
|---------------------|---------------------|
| Using ML to constrain ESM projections | ✅ Nature Comm 2022 wildfire |
| Emergent constraints methodology | ✅ Nature Comm 2022 wildfire |
| Multi-model ensemble analysis | ✅ Nature Comm 2022 wildfire |
| Observational data quality assessment | ✅ Nature Comm 2022 wildfire |
| Extreme event attribution | ⏳ To be added |
| XAI method validation | ⏳ To be added |
| Downscaling/super-resolution | ⏳ To be added |

---

**Your peer review exemplar knowledge base is ready. Add more Nature PDFs and continue building your collection!**