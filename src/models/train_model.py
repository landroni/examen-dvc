
import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

print(f"joblib version: {joblib.__version__}")

def train_model(processed_path = "data/processed"):
    ##import processed data
    X_train_scaled = pd.read_csv('data/processed/X_train_scaled.csv')
    X_test_scaled = pd.read_csv('data/processed/X_test_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')


    ##GridSearch ElasticNet
    reg_en = ElasticNet()
    params_grid_reg_en = {
        'l1_ratio': np.arange(0.0, 1.01, 0.05),
        'alpha': [0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    }
    grid_reg_en = GridSearchCV(cv=3, estimator=reg_en, param_grid=params_grid_reg_en, 
                            scoring='neg_mean_squared_error')
    grid_reg_en_fit = grid_reg_en.fit(X_train_scaled, y_train)

    ##best model
    grid_reg_en_best_params = grid_reg_en.best_params_
    print(pd.DataFrame.from_dict(grid_reg_en_fit.cv_results_).loc[:, ['params', 'mean_test_score']])
    print(grid_reg_en_best_params)


    #--Train the final model
    model_en = ElasticNet(alpha = grid_reg_en_best_params['alpha'], 
                        l1_ratio= grid_reg_en_best_params['l1_ratio'])
    model_en.fit(X_train_scaled, y_train)
    print(f"fitted model: {model_en}")
    
    #Save the trained model to a file
    model_filename = './models/trained_model.joblib'
    joblib.dump(model_en, model_filename)
    print(f"Model trained and saved successfully to {model_filename}.")


if __name__ == "__main__":
    train_model()