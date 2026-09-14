# LapTime-Predictor
Build an ML system that predicts a driver's next lap time using historical lap data, tyre information, weather conditions, and telemetry.

### Thoughts
First thing I was thinking was, it would be easier to build a model, and that too just for predicting numbers. 
But the F1 data needs precisions, that too in miliseconds. 

### Models
1. Built a simple linear regression, cause I haven't written a single line since 1.5 months. (Just for revision)
2. Built 2 catboost models. (one for race and another for qualifying) 
The qualifying performs better, but the race goes beyond acceptance.

Give it a try and try getting a the laptimes close to milliseconds.
