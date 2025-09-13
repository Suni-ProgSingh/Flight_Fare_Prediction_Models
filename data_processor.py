"""
Simple data processing utilities for the flight price prediction app
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def process_flight_data(df):
    """
    Process raw flight data similar to your notebook
    This is a simplified version for demo purposes
    """
    # handle date of journey
    df['Date'] = df['Date_of_Journey'].str.split('/').str[0].astype(int)
    df['Month'] = df['Date_of_Journey'].str.split('/').str[1].astype(int)
    df['Year'] = df['Date_of_Journey'].str.split('/').str[2].astype(int)
    
    # Handle Arrival Time
    df['Arrival_Time'] = df['Arrival_Time'].apply(lambda x: x.split(' ')[0])
    df['Arrival_Hour'] = df['Arrival_Time'].str.split(':').str[0].astype(int)
    df['Arrival_Minutes'] = df['Arrival_Time'].str.split(':').str[1].astype(int)
    
    # Handle Departure Time
    df['Dep_Hour'] = df['Dep_Time'].str.split(':').str[0].astype(int)
    df['Dep_Min'] = df['Dep_Time'].str.split(':').str[1].astype(int)
    
    # Handle Total Stops
    stop_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['Total_Stops'] = df['Total_Stops'].map(stop_mapping).fillna(1)
    
    # Handle Duration
    df['Duration_Hour'] = df['Duration'].str.split(' ').str[0].str.split('h').str[0]
    df['Duration_Min'] = df['Duration'].str.split(' ').str[1].str.split('m').str[0].fillna(0)
    df['DurationInMin'] = df['Duration_Hour'].astype(int) * 60 + df['Duration_Min'].astype(int)
    
    # Drop unnecessary columns
    columns_to_drop = ['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 
                      'Route', 'Duration_Hour', 'Duration_Min']
    df = df.drop(columns_to_drop, axis=1)
    
    return df

def encode_categorical_features(df, categorical_cols):
    """
    One-hot encode categorical features
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    encoded_df.index = df.index
    
    # Combine with other features
    df_processed = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    
    return df_processed, encoder
