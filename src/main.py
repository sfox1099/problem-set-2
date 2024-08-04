#'''
#You will run this problem set from main.py, so set things up accordingly
#'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot
import pyarrow.feather as feather
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
    arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw.filing_date)
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw.filing_date)
    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)

    # Save both data frames to `data/` -> 'pred_universe_raw.csv', 'arrest_events_raw.csv'
    pred_universe_raw.to_csv("C:\\Users\\sfox1\\414 PS2\\problem-set-2\\src\\Data\\pred_universe_raw.csv", index=False)
    arrest_events_raw.to_csv("C:\\Users\\sfox1\\414 PS2\\problem-set-2\\src\\Data\\arrest_events_raw.csv", index=False)



    # PART 2: Call functions/instanciate objects from preprocessing
    
    # Load the CSV files into dataframes
    pred_universe_raw = pd.read_csv("C:\\Users\\sfox1\\414 PS2\\problem-set-2\\src\\Data\\pred_universe_raw.csv")
    arrest_events_raw = pd.read_csv("C:\\Users\\sfox1\\414 PS2\\problem-set-2\\src\\Data\\arrest_events_raw.csv")

    # Convert the arrest_date_event and arrest_date_univ to datetime
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['arrest_date_event'], errors='coerce')
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['arrest_date_univ'], errors='coerce')

    # Join the datasets on person_id
    df_arrests = pd.merge(pred_universe_raw, arrest_events_raw, on='person_id', how='outer')
    print(df_arrests)

    # Create the 'y' column
    df_arrests['y'] = 0

    # Iterate through each row to check for felony arrests in the next year
    for index, row in df_arrests.iterrows():
        if pd.notna(row['arrest_date_univ']):  # Ensure there is a valid arrest date
            one_year_after = row['arrest_date_univ'] + pd.DateOffset(years=1)
            # Check for any felony arrest in the within 1 year
            felony_arrest = arrest_events_raw[
                (arrest_events_raw['person_id'] == row['person_id']) &
                (arrest_events_raw['arrest_date_event'] > row['arrest_date_univ']) &
                (arrest_events_raw['arrest_date_event'] <= one_year_after) &
                (arrest_events_raw['charge_degree'] == 'felony')  # Ensure to match the correct column
            ]
            if not felony_arrest.empty:
                df_arrests.at[index, 'y'] = 1

    # Calculate the share of arrestees that were rearrested
    share_rearrested = df_arrests['y'].mean()
    print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year? {share_rearrested:.2%}")

    # Create the current_charge_felony column
    df_arrests['current_charge_felony'] = (df_arrests['charge_degree'] == 'felony').astype(int)
    print(df_arrests)

    # Calculate the share of current charges that are felonies
    share_current_felony = df_arrests['current_charge_felony'].mean()
    print(f"What share of current charges are felonies? {share_current_felony:.2%}")

    # Create the num_fel_arrests_last_year column
    df_arrests['num_fel_arrests_last_year'] = 0

    for index, row in df_arrests.iterrows():
        if pd.notna(row['arrest_date_univ']):
            one_year_before = row['arrest_date_univ'] - pd.DateOffset(years=1)
            felony_arrest_count = arrest_events_raw[
                (arrest_events_raw['person_id'] == row['person_id']) &
                (arrest_events_raw['arrest_date_event'] > one_year_before) &
                (arrest_events_raw['arrest_date_event'] < row['arrest_date_univ']) &
                (arrest_events_raw['charge_degree'] == 'felony')
            ].shape[0]
            df_arrests.at[index, 'num_fel_arrests_last_year'] = felony_arrest_count

    # Calculate and print the average number of felony arrests in the last year
    average_fel_arrests_last_year = df_arrests['num_fel_arrests_last_year'].mean()
    print(f"What is the average number of felony arrests in the last year? {average_fel_arrests_last_year:.2f}")

    # Print the mean of num_fel_arrests_last_year
    print(f"Mean of num_fel_arrests_last_year: {df_arrests['num_fel_arrests_last_year'].mean()}")

    # Print the first few rows of df_arrests
    print(df_arrests.head())

    # Save df_arrests to a CSV file
    df_arrests.to_csv("C:\\Users\\sfox1\\414 PS2\\problem-set-2\\src\\Data\\df_arrests.csv", index=False)



    # PART 3: Call functions/instanciate objects from logistic_regression
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



    # PART 4: Call functions/instanciate objects from decision_tree




    # PART 5: Call functions/instanciate objects from calibration_plot


if __name__ == "__main__":
    main()

