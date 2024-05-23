# US 2024 Presidential Election Predictions

This repository contains the code and data for a project that aims to predict the outcome of the US 2024 presidential elections. Below is an explanation of the model, the data used, and how to interpret the results.

## Overview

This project showcases a half-serious attempt to predict the outcome of the US 2024 presidential elections. The reliability of polls with regard to elections is a topic of recurring debate. This is also the case in this election cycle, in which many polls show shifts in the electorate that seem unrealistic to some observers, such as a strong gain for Trump among young voters. While I don't feel qualified to judge the extent to which this is true, this model is based on the assumption that there is something to it and that polls alone are not enough to give us a realistic picture of the election outcome (otherwise there would be no need to make a model like this and I wanted to make a model, so here we are). Overall, this model is more of an experiment than a truly mathematically sound endeavor, so please keep that in mind.

[A overview of results can be found here](https://2024-election-predictions.tiiny.site/)

## Methodology

In addition to current polls for the presidential election, this model takes into account:

- **Polls for upcoming Senate, House, and gubernatorial elections:** Based on the assumption that voters who clearly favor one party here are also more likely to stick with that party in the presidential election.
- **Election results for presidential, Senate, House, and gubernatorial elections:** Based on the analogous assumption for past election results.
- **Demographic data from the 2020 Census:** Specifically age, ethnicity, education, population density, and proportion of residents in large cities, based on the assumption that these factors are useful for predicting how a state will vote.
- **Allan Lichtman's and Vladimir Keilis-Borok's "Keys to the White House":** Based on the assumption that the keys identified here are indeed suitable as predictive indicators.

All these factors are weighted based on values that have been automatically calculated by calibrating with some safe states and only slightly adjusted manually afterward. 

Special thanks to the following sources for providing the data used in this model:

- FiveThirtyEight for providing polls
- CORGIS for the census data
- Plotly for city population data 
- PublicaMundi for US state GeoJSON
(all via GitHub) 

## Contributing

If you have any feedback regarding the model, you can email me at [tim.menzner@hs-coburg.de](mailto:tim.menzner@hs-coburg.de).

