To run this code pip install numpy, pandas, scikit-learn, matplotlib, seaborn, ucimlrepo
Report for linear regression without library (Part 1):
    Trial 1 - No normalization, learn rate = 0.001: MSE: NaN, R^2: NaN
    After trial 1, I realized I had to normalize the features in order to generate legible resuslts.
    Trial 2 - learn rate = 0.001: MSE: 76.93764533516749, R² Score:  0.6602521295115495
    Trial 3 - learn rate = 0.0001: 279.134490371448 R² Score: -0.23262608662433237
    Trial 4 - learn rate = 0.01: MSE: 77.32758844234128 R² Score: 0.6585301852059134
    Trial 5 - learn rate = 0.1: MSE: 77.32837199634551 R² Score: 0.6585267251207603
    6. Answer this question: Are you satisfied that you have found the best solution? Explain.
        I am partially satisfied that I have found the best solution given the dataset and the knowledge on linear regression that we have learned so far. At first, I did not feel satisfied that I could not reduce the MSE to be less than 76.9.
        After researching and trying multiple different learn rates I realized that this was the best optimization I was going to get with this dataset and with the amount of linear regression I have learned so far. I was able to implement more advanced 
        features like normalization to improve my results but even with that this was the best result I was able to get with the data in respect to the MSE. I was more satisfied with the R^2 value and overall am satisfied for the first time trying linear regression.
    Plot attatched as LR_without_lib_plot.png
Report for SGDRegressor (Part 2):
    Trial 1 - learn rate = 0.001: MSE: 76.34438594536013 R² Score: 0.6591186346180187
    Trial 2 - learn rate = 0.0001: MSE: 76.1643162890059 R² Score: 0.6599226543185088
    Trial 3 - learn rate = 0.01: 75.45386374901045 R² Score: 0.6630948591751447
    Trial 4 - learn rate = 0.1: MSE: 82.46968662680628 R² Score: 0.6317688716484019
    Trial 5 - learn rate = 0.007 MSE: 75.34673855570831 R² Score: 0.6635731783299517
    6. Answer this question: Are you satisfied that you have found the best solution? Explain.
    I am satisfied that I have found the best solution given the dataset provided and the libraries implemented.
    I am much more satisfied by the consistency of this solution and am more satisfied with my linear regresion without libraries
    solution as well now because the results from both solutions were very similar.
    Plot attatched as SGDRegressor_plot.png