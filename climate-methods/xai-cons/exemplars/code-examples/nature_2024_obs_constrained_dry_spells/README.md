# EC4CDD_CMIPmodels
This repository contains Python codes to reproduce the main figures of the paper: "Observation-constrained projections reveal longer-than-expected dry spells" by Petrova I.Y. et al. 2024 (DOI: TBI)

Author: Irina Y. Petrova

## Abstract

Climate models project an exacerbation of dry extremes in many world regions 1,2⁠⁠⁠. However, confidence in the magnitude and timing of projected changes remains low 3,4⁠, leaving societies largely unprepared 5,6⁠. Here we show that constraining model projections with observations using a newly proposed emergent constraint (EC) reduces the uncertainty in future predictions of a core drought indicator, the longest annual dry spell (LAD), by 10–26% globally. The EC-corrected projections reveal a 42–44% higher LAD increase, on average, than “mid-range” or “high-end” future forcing scenarios suggest. Our results imply that the end-of-century global mean land-only LAD could be 10 days longer than currently expected. Using two generations of climate models, we further uncover global regions where historical LAD biases affect the magnitude of projected LAD increases, and explore the role of land–atmosphere feedbacks therein. Our findings expose regions with potentially higher- and earlier-than-expected drought risks for societies and ecosystems, and they point to possible mechanisms underlying the biases in the current generation of climate models. 


## Content 
* _ECpaper_Figure*py_ files reproduce the corresponding main figures of the paper
* _Map2_Corr.py_ and _useful_functions.py_ are aiding files for data processing and plotting
* EC_KL_div*py files are part of the Emergent Constrain calculation package adopted for this study. The full EC package can be found at https://doi.org/10.5281/zenodo.10886174 and Brient2020: https://doi.org/10.1007/s00376-019-9140-8

## Input data

Data required to reproduce the figures are available from https://doi.org/10.5281/zenodo.11636527

## System specifications

_This Python package has been prepared and run under the following specifications_
* Operating System (OS): GNU/ Linux x86_64
* Software: Conda (version = 4.10.3) 
* Software: Python (version = 3.7.13)

## Reference

I.Y. Petrova, Diego G.M., Florent B., Markus G.D., Seung-Ki M., Yeon-Hee K., Margot B. Observation-constrained projections reveal longer-than-expected dry spells (2024) Journal TBI

