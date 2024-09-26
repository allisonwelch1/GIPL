Author: Allison Welch, 2024
# GIPL (Geophysical Institute Permafrost Laboratory) Model

See the master branch for model documentation.

This branch was developed for single-site parameterization and for use in climate change experiments.

<h2>Study Site</h2>
<h5>Sagwon Bluffs, North Slope of Alaska, USA</h5>
Latitude: 69.418\
Longitude: -148.615\
SE facing slope above the floodplain of the Sagavanirktok River\
Arctic shrubland tundra\
Mean Annual Temperature: \
Mean Annual Precipitation: 133 mm/yr (Drew et al., 2023; Harris et al., 2020)

<h2>Observational Data</h2>
/calibration_run holds observational data (/data). 'Current_SAG_SNOWTEL..' contains
air temperature (C), snow depth (m), and snow thermal conductivity. Air temperature and snow depth
come from the Sagwon SNOTEL site (69 deg; 25 min N, 148 deg; 42 min W). Snow thermal conductivity is assumed
based on known values. 'mesres.txt' contains ground temperature data from the SagMAT (Moist Acidic Tundra) Arctic
Observatory Network (AON) monitoring site. The data is for the 2015 calendar year.

<h2>Workflow</h2>
<h3>1. Calibration</h3>
GIPL was parameterized using observational data and a sensitivity analysis (sa_for_gipl.py).
Visualization and RMSE calculation can be found in calibration_run/allison_results_plots.ipynb

<h3>2. Control Model Spin-Up</h3>
The calibrated model was spun up with 20 years of observational data (year 2015 x 20) until equilibrium was reached–/control_spinup. Model stability was
defined as the point when the maximum difference of soil temperatures at any depth between two successive annual cycles
was <0.1 C (equilibrium.ipynb). Temperatures at depth on the last day of the year after the first year equilibrium were used as initial conditions (/in/initial.txt)
for the control model.

<h3>3. Control Model (10 yrs)</h3>
After spin-up, the control model was run for 10 years (2015 x 10) for comparison against experimental runs
(/control_model_10yrs) with initial conditions from spin-up.

<h3>4. Control Model (1 yr)</h3>
After spin-up, the control model was run for 1 year for comparison against experimental equilibrium conditions
(/control_stable_yr) with initial conditions from spin-up.

<h3>Experimental Runs</h3>
For experimental runs, the experimental condition was applied to a 10-year model run. E.g., for 1 degree C of warming,
observational air temperature was modified to increase the mean annual temperature by ~1 C by year 10. Then steps 2–4 were repeated
for every experiment (spin-up to equilibrium (/_experiment_/spin_up), equilibrium state 1 year model with initial.txt from spin-up (/_experiment_/stable_yr)).

<h4>warming_experiment_1C</h4>
This experiment increases air temperature by 1 degree C over 10 years. Modified temperature (including induced heat waves)
was produced using /warming_experiment_1C/increase_air_temp.ipynb. /compare_control_warming1C.ipynb produces visualizations
and statistics that compare the 10-year control run and the 10-year experimental run. /compare_cont_exp_stable_years.ipynb
produces visualizations and statistics that compare the single-year equilibrium states.

<h4>warming_experiment_2C</h4>
Same as warming_experiment_1C, but with 2 degree C warming.



<h2>Run</h2>
To run an experiment, cd into that folder and run
```bash
~/GIPL/gipl config --gipl_config.cfg
```
Check that the input and outputs in the /_experiment_/gipl_config.cfg file are correct.

<h3>References</h3>
Drew, J. W., Bret-Harte, M. S., Buchwal, A., & Heslop, C. (2023). Age matters: Older Alnus viridis ssp. fruticosa are more sensitive to summer temperatures in the Alaskan Arctic. Functional Ecology, 37, 1463–1475. https://doi.org/10.1111/1365-2435.14307 

Harris, I., Osborn, T. J., Jones, P., & Lister, D. (2020). Version 4 of the CRU TS monthly high-resolution gridded multivariate climate dataset. Scientific Data, 7(1), 109.




