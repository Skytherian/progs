## [Ensemble Model](model2.ipynb)  
This model is created by making an ensemble model for non-linearly related factors we have used conditional inference trees and decision trees, and then used a WLS regression model with interaction effects after Box-Cox transformation.

## [SVM](svm.ipynb)
This model uses an SVM with radial kernel for regression 

## [XGBoost](xgboost.ipynb)
This model uses XGBoost technique for regression
   
#Comparision  
  

| Technique | Testing RMSE  |
| :-----:   | :-----:|
| Ensemble  | 10.267 |
| SVM       | 17.118 |
| XGBoost   | 9.81   |

## Conclusion
  
The XGBoost technique is the best for accuracy but the Ensemble model is not for behind, while the XGBoost model is not interpretable as much as the Ensemble model.  
  
From the diaganostic plots for the WLS model in the Ensemble model we can see that there is hetereoskedasticity present.
