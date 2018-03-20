import numpy as np
import pandas as pd

def load_bank_full(filename, outputfile='../output/bank_full_renamed.csv'):
    """
    Read the bank_full.csv file into a Pandas DataFrame and rename the columns.
    Stash the renamed Dataframe to a new file.
    Extract the target and return the features, target as Pandas DataFrame, Series
    
    Parameters
    ----------
    filename:       String, path of the bank-full.csv file
    
    outputfile:     String, path of the temporary file to store renamed DataFrame
    
    Returns
    --------
    Pandas DataFrame, Pandas Series
    
    """
    df = pd.read_csv(filename, ';')
    df.columns=['age', 'occupation', 'maritalStatus', 'education', 'hasCreditDefault', 'balance', 'hasHousingLoan', 'hasPersonalLoan',
                'contactMethod', 'contactDay', 'contactMonth', 'contactDurationInSec', 'numContactsPerformed','daysSinceLastContact',
                'numPrevContact', 'prevOutcome', 'target']
    
    df.to_csv(outputfile, index=False)
    
    return df

def get_year_and_weekday(df_day_month):
    """
    Use the information that the data runs from May 2008 to Nov 2010,
    add the year and day of week for every entry.
    
    Parameters
    -----------
    df_day_month:   Pandas DataFrame with columns 'contactDay' and 'contactMonth'
    
    Returns
    -------
    df:    Pandas DataFrame with columns 'contactYear' and 'weekDay'
    """
    df = df_day_month
    
    # Ugly hack to get these three years correct
    year = np.ones(len(df))*2008

    idx = df[pd.Index(df['contactMonth']).get_loc('jan')].index[0]
    year[idx:] = 2009
    new_df = df.iloc[idx:, :]
    idx2 = new_df[pd.Index(new_df['contactMonth']).get_loc('feb')].index[0]
    new_df2 = df.iloc[idx2:, :]
    idx3 = new_df2[pd.Index(new_df2['contactMonth']).get_loc('jan')].index[0]
    year[idx3:] = 2010
    
    # Calculate day of the week
    df['contactYear'] = year.astype(int).astype(str)
    
    months = {'jan':'01', 'feb':'02', 'mar':'03', 'apr':'04', 'may':'05', 'jun':'06', 
             'jul':'07', 'aug':'08', 'sep':'09', 'oct':'10', 'nov':'11', 'dec':'12'}
    df['monthString'] = df['contactMonth'].apply(lambda x: months[x])

    days = list(df['contactDay'].values.astype(str))
    days = [day.zfill(2) for day in days]
    df['dayString'] = days

    df['date'] = df['contactYear'] + df['monthString'] + df['dayString']
    df['weekDay'] = pd.to_datetime(df['date']).dt.weekday_name
    
    return df.loc[:, ['contactYear', 'weekDay']]


def extract_features(features):
    """
    Extract features from dataframe. 
    This includes: 
    1. Removing the 'contactDurationInSec' feature (not used in our model)
    2. Replace -1 in the 'daysSinceLastContact' feature (not contacted before) with a large value
    3. Convert the numeric features 'age', 'numContactsPerformed', 'numPrevContact', 'daysSinceLastContact' 
       to categorical features by binning them
   
    4. Add features 'yearOfContact' and 'dayOfWeek'
    
    Parameters
    ----------
    features_df: Pandas DataFrame
    
    Returns
    -------
    Pandas DataFrame with transformed features
    """
    features_df = pd.DataFrame.copy(features)
    
    features_df.drop('contactDurationInSec', axis=1, inplace=True)
    
    
    age_bins = [18, 30, 40, 50, 60, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '>60']
    age_binned = pd.cut(features_df['age'], bins=age_bins, labels=labels)
    
    n_contacts_bins = [0, 1, 2, 5, 100]
    labels = ['1', '2', '3-5', '>5']
    n_contacts_binned = pd.cut(features_df['numContactsPerformed'], bins=n_contacts_bins, labels=labels)
    
    n_prev_contacts_bins = [-1, 0, 1, 2, 5, 100]
    labels = ['0', '1', '2', '3-5', '>5']
    n_prev_contacts_binned = pd.cut(features_df['numPrevContact'], bins=n_prev_contacts_bins, labels=labels)
    
    n_days_bins = [0, 30, 90, 150, 210, 270, 330, 390, 450, 10000]
    labels = ['0-30', '30-90', '90-150',  '150-210', '210-270', '270-330', '330-390', '390-450', ">450"]
    n_days_since_contact_binned = pd.cut(features_df['daysSinceLastContact'], bins=n_days_bins, labels=labels)
    
    # Drop the numeric features and replace with binned features
    features_df.drop(['age', 'daysSinceLastContact', 'numContactsPerformed', 'numPrevContact'], axis=1, inplace=True)
    
    features_df['ageBinned'] = age_binned
    features_df['nContactsBinned'] = n_contacts_binned
    features_df['nPrevContactsBinned'] = n_prev_contacts_binned
    features_df['nDaysSinceLastContactBinned'] = n_days_since_contact_binned
    
    
    # Add year and day of Week information
    df_year_wd = get_year_and_weekday(features_df.loc[:, ['contactDay', 'contactMonth']])
    features_df = pd.concat([features_df, df_year_wd], axis=1)
    
    features_df['contactDay'] = features_df['contactDay'].astype(str)
    
    # Map target to 0, 1
    numeric = {'yes':1, 'no':0}
    features_df['target'] = features_df['target'].apply(lambda x: numeric[x])
    
    return features_df


def one_hot_encode_categorical(features):
    """
    One-hot encoding of categorical features for logistic regression and trees.
    Uses the Pandas built-in get_dummies method.
    """

    features['contactDay'] = features['contactDay'].astype('category')
    return pd.get_dummies(features)


def extract_features_target(input_df):
    """
    Extract target from DataFrame, convert the categorical features to one-hot vectors. 
    Return features and target as Pandas DataFrame and numpy array separately.
    
    Return
    ------
    features_encoded: Pandas DataFrame of one-hot encoded features
    
    target: Numpy array of target values (0 or 1)
    
    features_names: Array of string of the encoded feature names
    
    """
    df = pd.DataFrame.copy(input_df)
    
    target = df['target']
    del df['target']
    
    df.drop(['contactYear'], axis=1, inplace=True)
    
    features_encoded = one_hot_encode_categorical(df)
    
    # PrevOutcome_unknown is identical to nPrevContactsBinned_0, so remove one.
    features_encoded.rename(columns={'nPrevContactsBinned_0': 'noPrevContact'}, inplace=True)
    features_encoded.drop(['prevOutcome_unknown'], axis=1, inplace=True)
    
    
    return features_encoded, target.values.ravel()


