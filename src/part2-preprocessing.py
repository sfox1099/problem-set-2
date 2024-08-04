'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd

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