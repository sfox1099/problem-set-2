'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Your code here

# Read in "df_arrests"
df_arrests = pd.read_csv("C:\\Users\\sfox1\\414 PS2\\problem-set-2\\src\\Data\\df_arrests.csv")

# Split df_arrests into train and test dataframes
df_arrests_train, df_arrests_test = train_test_split(df_arrests, test_size=0.3, shuffle=True, stratify=df_arrests['y'])

# Create a list called features
features = ['num_fel_arrests_last_year', 'current_charge_felony']

# Create a parameter grid called param_grid
param_grid = {'C': [0.1, 1.0, 10.0]}

# Initialize the Logistic Regression model
lr_model = lr(solver='liblinear')

# Initialize the GridSearchCV
gs_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Run the model on the training data
X_train = df_arrests_train[features]
y_train = df_arrests_train['y']
gs_cv.fit(X_train, y_train)

# Optimal value for C and regularization information
optimal_C = gs_cv.best_params_['C']
print(f"Optimal value for C: {optimal_C}")

if optimal_C < 1:
    print("C has the most regularization (high regularization).")
elif optimal_C == 1:
    print("C has medium regularization.")
else:
    print("C has the least regularization (low regularization).")

# Predict for the test set
X_test = df_arrests_test[features]
df_arrests_test['pred_lr'] = gs_cv.predict(X_test)

# Save the test dataframe to a CSV
df_arrests_test.to_csv("C:\\Users\\sfox1\\414 PS2\\problem-set-2\\src\\Data\\df_arrests_test_with_predictions.csv", index=False)