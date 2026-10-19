## Alloy-718 Yield Strength Model

This repository contains the code accompanying the manuscript to estimate the
**room-temperature yield strength** of Alloy-718 type nickel superalloys from
CALPHAD-derived phase fraction data. For each alloy composition, four
strengthening contributions are calculated and added together.

### Details of v1

The yield strength is obtained as the sum of the Hall–Petch, solid-solution and
precipitate contributions:

$$
\sigma_{YS} = \sigma_{HP} + \sigma_{SS} + \sigma_{ppt}
$$

where the precipitate contribution is the sum of the γ′ and γ″ terms:

$$
\sigma_{ppt} = \sigma_{oro,\gamma'} + \sigma_{coh,\gamma''}
$$

| Contribution | Mechanism |
|---|---|
| $\sigma_{HP}$ | Hall–Petch (grain-boundary) strengthening for a fixed grain size |
| $\sigma_{SS}$ | Solid-solution strengthening of the γ matrix |
| $\sigma_{oro,\gamma'}$ | Orowan strengthening from γ′ precipitates |
| $\sigma_{coh,\gamma''}$ | Coherency-strain strengthening from γ″ precipitates |

The output is a table listing each contribution and the total yield strength
for every composition.

### Details of v2

*To be added.*

> [!NOTE]
> No real data is included in this repository. When run without inputs, the
> script uses randomly generated placeholder data to demonstrate the workflow.

## Method

For full details of the method, please refer to the **Supplementary Information**
of the manuscript.
