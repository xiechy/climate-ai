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

#### 1. Observation-Constrained Predictions of First Arctic Ice-Free Day  
- **Journal:** Nature Communications  
- **Year:** 2024
- **Paper Title:** "The first ice-free day in the Arctic Ocean could occur before 2030"
- **PDF Filename:** `nat_commun_2024_arctic_ice_free_day.pdf`  
- **Topic:** Daily-resolution Arctic sea ice predictions using CMIP6 models with observation-based model selection and internal variability quantification

**Key Strengths of Review:**
- Systematic framework for separating internal variability from forced response in near-term climate predictions
- Three reviewers with complementary expertise: climate dynamics (R1), sea ice physics and observations (R2), polar systems and science communication (R3)
- Rigorous requirements for validating model-predicted rapid transitions against observed variability
- Strong emphasis on scientific communication: balancing scientific rigor with public understanding
- Two-round review demonstrating successful manuscript evolution from narrow framing to broader impact

**Notable Sections:**

**Internal Variability vs. Forced Response (Reviewers 1 & 2):**
- Questioned whether 7% probability (from counting CMIP6 members) represents real probability given dependence on model selection and ensemble size
- Demanded distinction between earliest ice-free days driven by internal variability vs. those driven by emission scenarios
- Key insight: Fastest 3 transitions (3-4 years) occurred under SSP1-2.6 (second-lowest forcing), with rapid transitions under all scenarios except SSP1-1.9, proving internal variability dominance
- Required showing emission scenario irrelevance explicitly, not implicitly

**Observational Feasibility Assessment (Reviewers 1 & 2):**
- R2 challenged: "If maximum observed year-to-year SIA drop repeated annually, would that reach ice-free Arctic within 6 years? If model RILEs never occurred in observations, how confident can we be this isn't model-world issue?"
- Authors added analysis: only three 2011-2012 level ice loss events needed to transition from 2023 conditions to ice-free state
- Historical observation nearly qualified as RILE: 2004-2007 observed 5-yr running mean SIE trend reached -0.22 million km²/year (RILE threshold: -0.3 million km²/year)
- Demonstrated two extreme events (2006-2007 and 2011-2012 level drops) within 4-year period would have constituted RILE
- Conclusion: RILEs (Rapid Ice Loss Events) "entirely possible" given observed record

**Final-Year Trigger Analysis Rigor (Reviewer 1):**
- Original Section 2.2.2 criticized as "a bit anecdotal" - lacking comparison to "normal" years vs. just climatology
- Demanded: "Are final years more stormy than normal compared to other simulations? Authors should include more rigorous analysis"
- Authors added Extended Data Table 3: statistical tests showing final year significantly anomalous relative to preceding years in central Arctic and peripheral source regions
- Separated preconditioning analysis: September RILEs start mean 1.7 years before 2023-equivalent year (range: -4 to 0 years), median 1 year
- Found no relationship between longer preconditioning duration and faster transition speed

**Scientific Communication Framework (Reviewers 2 & 3):**
- R2: "High-impact part is not crossing the million km² threshold but rather getting a RILE causing extremely rapid drop in just a few years - this is choice of framing, not critique of results"
- R3 warned: "Challenging to publish with headline 'N% risk... before 2030' without being accused of fear-mongering... '7%' likely to produce 'meh' response for those unfamiliar with real consequences"
- R3: "First ice-free day is symbolic (arbitrary 1M km² limit), harbinger of things to come, but doesn't necessarily have immediate far-reaching consequences"
- Author response strategy:
  - Removed all specific percentages from title and text
  - New title: "The first ice-free day in the Arctic Ocean could occur before 2030" (removed "7% chance")
  - Abstract reframed: "highly symbolic event visibly demonstrating anthropogenic impact" rather than ecological/economic impacts
  - Conclusion added: "first ice-free day has symbolic significance but doesn't mean Arctic becomes ice-free every year... ice-free September doesn't imply ice-free year-round"

**Model Selection Methodology (Reviewer 2):**
- Challenged 14-year evaluation period: "large role of natural variability on such timescales (1/3 of sea ice loss can be due to natural variability) - are you just selecting models with right timing of internal variability?"
- Authors revised evaluation period from 14 years (2000-2014) to 20 years (1995-2014) to better handle internal variability
- Critical clarification added: "include models as long as at least one ensemble member matches criteria" - NOT selecting specific members matching observations
- This prevents selection bias toward specific internal variability phasing

**Physical Mechanism Verification (Reviewer 2):**
- "These models you've selected: where do they sit in winter negative feedback space? Do they get RILEs because greatly underestimated negative feedback in response to summer ice loss?"
- Authors demonstrated: quick transition simulations from 4 different models don't have special feedbacks because other ensemble members from same 4 models take much longer to reach ice-free conditions
- Cited Sticker et al. (2024): RILEs occur in almost all CMIP6 models, so not model-specific physics but internal variability in specific ensemble members

**Highlights of Author Response:**
- Calculated: 2004-2007 observed 5-yr running mean SIE trend fell only 0.08 million km²/year short of RILE qualification (8% below threshold)
- Two extreme September SIE loss events (2006-2007: -1.5 million km²; 2011-2012: -0.99 million km²) within 4-year period would have produced RILE
- Demonstrated RILEs are not model artifacts but observationally plausible
- Removed specific percentages while retaining message about possibility
- Changed from quantitative risk to qualitative possibility
- Figures 3 & 4 simplified: changed from individual simulations to multi-simulation averages
- Added "take-home message" to all figure captions
- Changed CDD (Cooling Degree Days, misleading) to HDD (Heating Degree Days)
- Added discussion of average preconditioning duration
- New Extended Data Table 3: statistical analysis showing final year anomalous

**Relevant for:**
- Climate predictions with observation-based model selection
- Internal variability vs. forced response separation
- Rapid transition event identification (RILEs in sea ice, applicable to other systems)
- Multi-model ensemble analysis with unequal ensemble sizes
- Science communication for low-probability high-impact events
- Arctic sea ice modeling and attribution

**Key Takeaways:**

**Prediction Methodology:**
1. **Daily vs. monthly data precedence**: First ice-free state observable in daily satellite data before monthly averages - requires daily-resolution analysis
2. **Internal variability dominates near-term**: Earliest ice-free days driven primarily by internal variability, not emission scenarios (fastest transitions under SSP1-2.6, second-lowest forcing)
3. **Multi-ensemble inclusion strategy**: Include all models with at least one member matching observations, not individual member selection, to avoid internal variability selection bias
4. **Evaluation period length critical**: Minimum 20-year evaluation period needed to reduce internal variability influence on model selection
5. **RILE event definition**: Rapid Ice Loss Events defined as 5-year running mean trend ≥0.3 million km²/year for September sea ice extent

**Observational Constraints and Validation:**
6. **Observational feasibility benchmarking**: Compare model predictions to historical extreme events (2007, 2012) to prove rapid transitions plausible, not model artifacts
7. **Threshold sensitivity quantification**: Historical observations came within 8% of RILE threshold, demonstrating such events aren't model-specific
8. **Preconditioning timing assessment**: September RILEs start mean 1.7 years before critical transition year (range: -4 to 0 years, median 1 year)
9. **Preconditioning duration ≠ transition speed**: Longer preconditioning doesn't necessarily cause faster ice loss - no correlation found
10. **Compound event requirement**: Two 2006-2007 or 2011-2012 level loss events within 4-year period would constitute RILE under current conditions

**Trigger Factor Identification:**
11. **Final-year anomaly statistical testing**: Must statistically test whether final year significantly anomalous relative to preceding years, not just vs. climatology
12. **Preconditioning vs. trigger separation**: Distinguish multi-year winter/spring preconditioning from final-year summer warm/stormy triggers
13. **Spatial source tracking**: Analyze peripheral seas (not just central Arctic) as storm source regions

**Uncertainty Quantification:**
14. **Model-dependence acknowledgment**: Probability estimates depend on model selection and ensemble size - should not be presented as precise predictions
15. **Feedback mechanism verification**: Check whether rapid-transition models have biased feedbacks by comparing to other ensemble members from same model
16. **Observed variability comparison**: Simulated RILE intensity should be benchmarked against observed maximum year-to-year variability

**Science Communication Strategies:**
17. **Symbolic vs. actual impact distinction**: First ice-free day ≠ consistently ice-free summers ≠ year-round ice-free - explicitly separate symbolic threshold from ecological/economic consequences
18. **Avoid precise probabilities for uncertain events**: Use "could occur" instead of specific percentages when probabilities depend on model sampling
19. **Emphasize pace of change over threshold crossing**: Focus on "rapid collapse within few years" rather than "crossing specific threshold"
20. **Explain scenarios in accessible terms**: SSP1-1.9 (CO₂ below current), SSP5-8.5 (high emissions) with endpoint CO₂ concentrations, not just numerical labels
21. **Guard against "fear-mongering" accusations**: Balance scientific rigor with public communication
22. **Provide physical context for non-specialists**: Help general readers understand what changes mean

**Manuscript Improvement Tactics:**
23. **Figure caption take-home messages**: Every figure caption must explicitly state what readers should see
24. **Multi-model averaging for clarity**: Use multi-model averages over individual model results when message doesn't require showing model diversity
25. **Terminology standardization**: Change abbreviations with multiple common meanings to field-standard usage
26. **Consistency checks**: Terms in figures must match text

---

#### 2. Combined Emergent Constraints on Future Extreme Precipitation Changes
- **Journal:** Nature Climate Change (Nature Portfolio)
- **Year:** 2024
- **Paper Title:** "Combined emergent constraints on future extreme precipitation changes"
- **PDF Filename:** `nat_clim_change_2024_extreme_precipitation_ec.pdf`
- **Topic:** Combined emergent constraints integrating global warming and extreme precipitation sensitivity to reduce uncertainty in CMIP5/6 projections

**Key Strengths of Review:**
- Two-round review by two methodologically rigorous reviewers (EC methodology expert + climate scientist)
- Systematic addressing of overconfidence in combined constraints (following Bretherton & Caldwell 2020 framework)
- Deep scrutiny of baseline period choice and its physical justification (1851-1900 vs. recent past)
- Strong emphasis on quantitative reporting and spatial pattern robustness validation
- Comprehensive treatment of internal variability across scales (global mean vs. regional grid points)
- Constructive evolution: manuscript significantly improved with added quantification, interpretation, and sensitivity tests

**Notable Sections:**

**Overconfidence Prevention (Reviewer 1):**
- Core concern: Combining constraints only useful if they provide independent information
- Required explicit accounting for correlation (r₀=0.28) between ΔTgm and ΔRgm/ΔTgm
- If correlation ignored: RRV would be overestimated at 51% instead of actual 42%
- Regional analysis: ignoring ΔR/ΔTgm vs. ΔTgm correlations → 5-25% RRV overestimation
- Authors incorporated Eq. 9 and 11 covariance terms to avoid overconfident uncertainty reduction
- Demonstrated method's sensitivity by showing what happens with r₀=0 assumption

**Baseline Period Justification (Reviewers 1 & 2):**
- R1 challenged 1851-1900 baseline: "portion of change already occurred, making y-axis not just future change"
- Authors' defense: ΔTgm-related EC requires preindustrial baseline because:
  - Both 2051-2100 and 1851-1900 have minimal aerosol forcing
  - trTgm (1970-2022 trend) driven mainly by GHGs, not aerosols
  - Recent baseline would include aerosol decline response, requiring separate constraint method
- Sensitivity test with 1970-2022 baseline showed RRV reduction: 42%→41% (combined EC)
- Authors demonstrated global mean Rgm values similar between 1997-2019 and 1851-1900 (Supplementary Fig. 1)
- Regional correspondence also shown: differences between periods are small

**Internal Variability Treatment (Reviewer 2):**
- R2 concern: "Grid-point scale internal variability in precipitation could be large, not negligible like global mean"
- Authors' response:
  - Showed spatial maps of internal variability vs. inter-observational-dataset variance (Supplementary Fig. 7)
  - Most regions: observational dataset differences dominate over internal variability
  - Note: 1997-2019 averaging reduces internal variability contribution
  - Acknowledged observational uncertainty sources: station coverage, satellite algorithms, data assimilation, bias correction

**Spatial Pattern Robustness (Reviewer 1):**
- Demanded CMIP5/6 split validation: "stipple areas with significant correlations in both"
- Authors created Supplementary Fig. 4: separate correlation maps for CMIP5-only, CMIP6-only, and combined
- ΔR vs. trTgm correlations similar between CMIP5/6, but CMIP6 shows larger areas of significance (more "hot models")
- ΔR/ΔTgm vs. R correlations consistent across both ensembles
- Physical explanation: regional temperature and humidity changes well-related to ΔTgm globally (Shiogama et al. 2024)

**Quantification Requirements (Reviewer 1 - 14 specific requests):**
- Original manuscript lacked concrete numbers in main text
- Authors added:
  - Table 1 & Supplementary Table 2: unconstrained vs. constrained estimates with percentiles
  - Specific variance reduction percentages (8-30%)
  - Central estimates with ±1 SD bounds (e.g., ΔRgm/ΔTgm: 2.07 [0.316, 3.82] raw → 2.03 [0.632, 3.44] constrained)
  - Correlation coefficients explicitly stated (0.55, 0.76)
  - Standard errors of mean for observational datasets
  - Spatial area percentages for RRV thresholds

**Method Clarity (Reviewer 2):**
- R2 confused about "hierarchical EC" vs. "combined EC" terminology
- Authors clarified:
  - Hierarchical EC framework (Bowman et al. 2018) used for all single-variable constraints (Figs. 1a-c)
  - Combined EC methods integrate information from multiple constrained variables
  - ΔR/ΔTgm-related EC: uses constrained ΔRgm/ΔTgm + raw ΔTgm
  - Full combined EC: uses constrained ΔTgm + constrained ΔRgm/ΔTgm
- Added Supplementary Fig. 2: joint distribution visualization of the two constrained variables

**Outlier Sensitivity Testing (Reviewer 1):**
- R1 noted one model with Rgm≈80 mm/day (major outlier) affects model spread and RRV calculation
- Authors implemented "1 ESM removed test" (jackknife approach)
- Supplementary Fig. 3e & 6e: RRVs not sensitive to excluding any single model
- Demonstrated results robust to ESM sampling, not driven by outliers

**Regional EC Physical Mechanisms (Reviewer 1, expanded in revision):**
- Why ΔTgm-related EC works regionally: thermodynamic components of ΔR linked to regional T and humidity changes
- Why ΔR/ΔTgm-related EC effective: precipitation sensitivity to humidity critical worldwide (ITCZ, storm tracks, monsoons)
- Complementary effectiveness:
  - ΔTgm-related EC: weak in tropical oceans (large dynamic component)
  - ΔR/ΔTgm-related EC: strong in major precipitation regions
  - Combined EC: both constraints reinforce in mid/high latitudes
- Added 38-line section (L247-285) explaining humidity's role

**Comparison with Other Studies (Reviewer 1):**
- Cannot quantitatively compare due to different metrics:
  - This study: absolute changes in Rx1day
  - Thackeray et al. 2022: ≥99th percentile precipitation
  - Paik et al. 2023 / Dai et al. 2024: percentage changes in Rx1day
  - Li et al. 2024: 50-year event intensity changes
- Authors elaborated on methodological comparisons (L307-328)

**Highlights of Author Response:**
- Moved humidity analysis from supplementary to main text (added ~40 lines)
- Created two new summary tables (Table 1, Supplementary Table 2)
- Added 7 new sensitivity analyses:
  - Doubled internal variability variance: results robust
  - 1970-2022 baseline: RRV reduced but method still effective
  - Jackknife ESM exclusion: not driven by outliers
  - CMIP5/6 split validation: spatial patterns consistent
  - Ignored r₀ scenario: quantified overconfidence risk
- Supplementary Fig. 7: spatial maps of observational uncertainty + internal variability
- Corrected minor GPCP value error (33.7→33.8 mm/day) and Figure 4f code bug (no result changes)
- Added detailed descriptions of observational datasets (GPCP, MSWEP2, GSWP3-W5E5)
- Fixed typo in revised version: "50 percentile" → "50th percentile"

**Relevant for:**
- Emergent constraints methodology development
- Combined constraint frameworks (Bretherton & Caldwell 2020 application)
- Extreme precipitation projection studies
- CMIP5/6 multi-model ensemble analysis
- Observational uncertainty quantification (precipitation products)
- Internal variability vs. forced response separation
- Baseline period selection for climate projections
- Regional application of global-scale constraints

**Key Takeaways:**

**Combined Constraint Methodology:**
1. **Correlation accounting essential**: Must explicitly incorporate covariance between constrained variables (e.g., ΔTgm vs. ΔR/ΔTgm) to avoid overconfidence
2. **Independent information criterion**: Following Bretherton & Caldwell (2020), combining constraints only valuable if they provide non-redundant information
3. **Quantify overconfidence risk**: Show sensitivity test of what RRV would be if correlations ignored (e.g., 51% vs. actual 42%)
4. **Complementary constraint design**: Best combined ECs target different uncertainty sources (thermodynamic vs. dynamic components)

**Baseline Period Selection:**
5. **Match baseline to forcing**: For GHG-driven changes, use preindustrial baseline; for total anthropogenic forcing, recent past acceptable
6. **Aerosol considerations**: If baseline includes significant aerosol forcing, need separate constraint for aerosol response
7. **Sensitivity test alternative baselines**: Show how RRV changes with different baseline choices (e.g., 1851-1900 vs. 1970-2022)
8. **Regional correspondence check**: Demonstrate climatological values similar between historical baseline and observational period

**Internal Variability Handling:**
9. **Scale-dependent treatment**: Global mean precipitation internal variability small; grid-point scale can be large
10. **Time averaging reduces variability**: Multi-decadal means (e.g., 1997-2019) reduce internal variability contribution
11. **Compare to observational uncertainty**: Show spatial maps of internal variability vs. inter-dataset variance
12. **Regional EC requires both**: Must quantify observational uncertainty + internal variability at grid scale

**Spatial Pattern Validation:**
13. **Ensemble split validation**: Separately analyze CMIP5 and CMIP6 to confirm pattern robustness
14. **Stipple dual significance**: Mark regions with significant correlations in both ensembles
15. **Physical mechanism explanation**: Connect regional patterns to known climate dynamics (e.g., thermodynamic scaling)
16. **Outlier sensitivity**: Jackknife test by excluding each model to verify spatial results not driven by single model

**Quantitative Reporting Standards:**
17. **Central estimate + uncertainty bounds**: Report 50th percentile with [2.5th, 97.5th] percentiles for raw and constrained
18. **Variance reduction quantification**: State RRV percentages explicitly in main text
19. **Effect size context**: Show constrained central estimate change relative to climatological value
20. **Correlation coefficients**: Display r values prominently (figure annotations + text)
21. **Observational product details**: Describe data sources, merging methods, uncertainty sources for each dataset

**Observational Data Quality:**
22. **Inter-dataset variance often dominates**: For precipitation, differences between products often larger than internal variability
23. **Uncertainty source taxonomy**: Station coverage, satellite algorithms, reanalysis methods, bias correction all contribute
24. **Standard error reporting**: Use SE=s/√n for climatologies, not just standard deviation
25. **Multiple product comparison**: Use ≥3 independent observational products to capture structural uncertainty

**Manuscript Evolution Strategy:**
26. **Add quantification progressively**: Original draft qualitative; revised version added ~15 specific numbers to main text
27. **Expand interpretation**: Move from "variance reduced" to "why this EC works in these regions"
28. **Supplement to main text migration**: Move key results from supplementary when reviewers request more detail
29. **Sensitivity analyses preemptive**: Authors added 7 robustness tests beyond what reviewers explicitly requested
30. **Fix minor errors transparently**: Corrected GPCP value and code bug, noted no impact on conclusions

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

- **Last Updated:** October 28, 2025
- **Total Reviews:** 3
- **Coverage Gaps:** Areas needing more examples
  - Paleoclimate modeling
  - Climate economics/IAMs
  - Regional climate modeling
  - Coupled model development
  - High-resolution process studies
  - Extreme event attribution (needed)
  - XAI method validation (needed)
  - Observational constraint methods (have 2, need more diversity in constraint types)  

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
| Emergent constraints methodology | ✅ Nature Comm 2022 wildfire, ✅ Nat Clim Change 2024 precipitation |
| Combined/multiple constraint frameworks | ✅ Nat Clim Change 2024 precipitation |
| Addressing overconfidence in constraints | ✅ Nat Clim Change 2024 precipitation |
| Baseline period selection for projections | ✅ Nat Clim Change 2024 precipitation |
| Multi-model ensemble analysis | ✅ Nature Comm 2022 wildfire, ✅ Nat Clim Change 2024 precipitation |
| Observational data quality assessment | ✅ Nature Comm 2022 wildfire, ✅ Nat Clim Change 2024 precipitation |
| Separating internal variability from forced response | ✅ Nature Comm 2024 Arctic, ✅ Nat Clim Change 2024 precipitation |
| Validating rapid transition predictions | ✅ Nature Comm 2024 Arctic |
| Science communication for climate risks | ✅ Nature Comm 2024 Arctic |
| Extreme precipitation projections | ✅ Nat Clim Change 2024 precipitation |
| Regional application of global constraints | ✅ Nat Clim Change 2024 precipitation |
| CMIP5/6 ensemble comparison | ✅ Nat Clim Change 2024 precipitation |
| Extreme event attribution | ⏳ To be added |
| XAI method validation | ⏳ To be added |

---

---

**Your peer review exemplar knowledge base is ready. Add more Nature PDFs and continue building your collection!**