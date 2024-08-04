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

    # PART 3: Call functions/instanciate objects from logistic_regression

    # PART 4: Call functions/instanciate objects from decision_tree

    # PART 5: Call functions/instanciate objects from calibration_plot


if __name__ == "__main__":
    main()

