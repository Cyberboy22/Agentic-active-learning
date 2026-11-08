# Can the expensive method ever pay for itself?

Landscapes chosen because co-folding is the *more accurate* model on them. It still costs eight docking runs per compound.

| target | budget | k | docking | co-folding | winner |
|---|---|---|---|---|---|
| egfr#1 | 240 | 10 | 0.633 | 0.167 | docking |
| egfr#1 | 240 | 50 | 0.393 | 0.087 | docking |
| egfr#1 | 960 | 10 | 0.367 | 1.000 | cofolding |
| egfr#1 | 960 | 50 | 0.347 | 0.527 | cofolding |
| cdk2#0 | 240 | 10 | 0.300 | 0.333 | cofolding |
| cdk2#0 | 240 | 50 | 0.400 | 0.173 | docking |
| cdk2#0 | 960 | 10 | 0.300 | 0.867 | cofolding |
| cdk2#0 | 960 | 50 | 0.313 | 0.500 | cofolding |

Co-folding won 5 of 8 settings.

Coverage, not accuracy, is what a recall-of-top-k metric rewards: putting k compounds in front of a chemist requires having measured about k of them, and an eight-times pricier method measures an eighth as many. Where co-folding wins, it is because k is small enough or the budget large enough that both methods can cover the shortlist and accuracy decides. Where it loses, no amount of accuracy compensates.