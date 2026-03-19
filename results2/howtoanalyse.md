How to analyse the data in this folder (results2)?

Compare with the data in:
- https://arc-aiaa-org.tudelft.idm.oclc.org/doi/abs/10.2514/6.2024-3520
Negative thrust investigation for a tip-mounted propeller rather than a tail-mounted propeller.
- https://ntrs.nasa.gov/api/citations/19710022893/downloads/19710022893.pdf
Negative thrust investigation for a front-mounted propeller rather than a tail-mounted propeller. (inludes info on control effectiveness)
- https://repository.tudelft.nl/file/File_4d4586dc-9de4-45d5-abc3-53b6f1b5ae68?preview=1
Investigaton on tail-mounted positive thrust propeller, can maybe be extrapolated / compared to the tail-mounted negative thrust propeller.
  
(You don't have to read the reports, just look at the figures)

Notes:
- eta_recuperation (efficiency of energy recuperation) is what is needed when C_P is negative (so energy is removed from
the flow such that it slows down).
In this case eta_recuperation is expected to be between 0 and 1.
- eta_propulsive (propulsive efficiency) is what is needed when C_P is positive (energy is used to accelerate the flow). 
In this case eta_propulsive is expected to be between 0 and 1.
- Write in the report about te relations between the variables, for example how CL seems to be independent of J, or how eta_recuperation is independent of AoA.
Also mention that at high J, C_T has a larger dependence on AoA than at low J, which is confirmed by fig 10 of the first source.
Also, higher AoA is associated with higher C_T for constant J, which is confirmed by fig 10 of the third source.
- The fit used for C_T and C_P is taken from appendix B of the manual, and is usually for positive thrust, so the fit does not
accurately capture the data.
- Say something about the destabilizing effect of negative thrust at some tail incidence angles i_t, as shown on page 55, 62, 76, 120, 121, etc. of the second source for example.
This effect is strongest on large negative dE (i.e. page 121 is more destabilizing than page 119). Watching at our data, you can see a small destabilizing
dCM_c4 only for the case dE=-20deg, for the other dE's the relation is very noisy. Overall, the propeller has only a small effect on the total Cm_c4 vs AoA of the aircraft, so 
the reverse thrust at our aircraft model with the given i_t is not very destabilizing.