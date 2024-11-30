# General purpose libraries
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Data visualization
import matplotlib.pyplot as plt

# Financial data handling
import yfinance as yf

# Machine learning models handling
import joblib


def create_daily_date_list(start_date:str, end_date:str)->list:
    """
    Input: start_date and end_date as strings in "YYYY-MM-DD" format
    Output: List of dates between start_date and end_date in "YYYY-MM-DD" format
    Logic: Iterates from start_date to end_date, adding each day to the list in "YYYY-MM-DD" format
    """

    
    # Parse the start and end dates from string format to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Initialize an empty list to store the dates
    date_list = []
    
    # Generate dates by iterating from start to end date with daily frequency
    current_date = start
    while current_date <= end:
        date_list.append(current_date.strftime("%Y-%m-%d"))  # Append date as a string in 'YYYY-MM-DD' format
        current_date += timedelta(days=1)  # Move to the next day

    return date_list

def get_last_friday(date_to_consider):
    """
    Input: No arguments; the function uses the current date as the starting point.
    Output: Returns `end_date` as a string formatted as `YYYY-MM-DD`, representing the last Friday's date.
    Logic: Calculate the number of days back to the last Friday.    
    
    """
    # Calculate the number of days back to the last Friday
    days_back = (date_to_consider.weekday() - 4) % 7
    last_friday = date_to_consider - timedelta(days=days_back)
    end_date = last_friday.strftime("%Y-%m-%d")
    return end_date


# Feature engineering for time series
def create_features(df):
    """
    Create time-based features for time series data.
    
    Input:
    - df: DataFrame with a datetime index and a single column for the target variable.

    Output:
    - df: DataFrame with added columns for year, month, week of the year, quarter, and cuatrimestre.

    Logic:
    - Extracts the 'year', 'month', 'week_of_year', 'quarter', and 'cuatrimestre' 
      from the datetime index to create additional time-based features.
    """
    df = df.copy()  # Create a copy of the DataFrame to avoid the warning
    df['year'] = df.index.year            # Extract year from the date
    df['month'] = df.index.month          # Extract month from the date
    df['week_of_year'] = df.index.isocalendar().week  # Extract week of the year from the date
    df['quarter'] = df.index.quarter      # Extract quarter from the date
    df['cuatrimestre'] = ((df.index.month - 1) // 4) + 1  # Calculate cuatrimestre (four-month period)
    return df

def get_historical_data(ticker_symbol:str, start_date:str, end_date:str)->pd.DataFrame:
    """
    Fetches historical stock data for a given company between specified dates. 
    Returns a DataFrame containing the historical data.

    Input:
    - ticker_symbol: The stock ticker of the company (e.g., 'ASML').
    - start_date: The start date for the historical data (e.g., '2000-01-01').
    - end_date: The end date for the historical data (e.g., '2024-10-05').

    Output:
    - A pandas DataFrame containing the historical stock data for the specified period.
    """
    # Define the ticker symbol
    stock_data = yf.Ticker(ticker_symbol)

    # Download historical data between the start and end dates
    historical_data = stock_data.history(start=start_date, end=end_date)

    # Reset the index to convert 'Date' from an index to a regular column
    historical_data.reset_index(inplace=True)

    # Return the resulting DataFrame
    return historical_data

def plot_closing_prices(data:pd.DataFrame, close_column:str, company_name:str):
    """
    Plots the historical closing prices of a company's stock over time.

    Input:
    - data: DataFrame containing the stock data with 'Date' and 'Close' columns.
    - close_column: The name of the 'Close' column to be displayed in the plot title.
    - company_name: The name of the company to be displayed in the plot title.

    Output:
    - A line plot of the company's historical closing prices.
    """
    
    # Convert the 'Date' and 'Close' columns to numpy arrays
    dates = np.array(data.index)
    close_prices = np.array(data[close_column])

    # Create the plot for the 'Close' (closing price) column using matplotlib
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(dates, close_prices, label='Closing Price', color='blue')

    # Add title and axis labels
    plt.title(f'Historical Closing Prices of {company_name} between {dates[0]} and {dates[-1]}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')

    # Rotate X-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the legend
    plt.legend()

    # Adjust layout to prevent overlap of elements
    plt.tight_layout()

    # Display the plot
    plt.show()
    
def plot_stock_with_multiple_boxplots(data: pd.DataFrame, close_column: str, company_name: str) -> pd.DataFrame:
    """
    Plots the stock's closing price distribution for weekly, monthly, and quarterly with
    multiple boxplots for each period.

    Input:
    - data: DataFrame containing the stock data with 'Date' and 'Close' columns.
    - close_column: The name of the 'Close' column to be displayed in the plot title.
    - company_name: The name of the company to be displayed in the plot title.

    Output:
    - Boxplots for each period (7 for weekly, 12 for monthly, and 4 for quarterly).
    """
    # Ensure 'Date' column is in datetime format only if it is not already
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'])

    # Extract additional time period information
    data['Weekday'] = data['Date'].dt.weekday  # Weekday (0=Monday, 6=Sunday)
    data['Month'] = data['Date'].dt.month      # Month (1=January, 12=December)
    data['Quarter'] = data['Date'].dt.quarter  # Quarter (1 to 4)

    # Plot Weekly Data (7 boxplots for each day of the week)
    plt.figure(figsize=(12, 8))
    data.boxplot(column=close_column, by='Weekday', grid=True)
    plt.title(f'{company_name} - Weekly Closing Prices (by Weekday)')
    plt.suptitle('')
    plt.xlabel('Weekday (0=Monday, 6=Sunday)')
    plt.ylabel('Closing Price (USD)')
    plt.show()

    # Plot Monthly Data (12 boxplots for each month)
    plt.figure(figsize=(12, 8))
    data.boxplot(column=close_column, by='Month', grid=True)
    plt.title(f'{company_name} - Monthly Closing Prices (by Month)')
    plt.suptitle('')
    plt.xlabel('Month')
    plt.ylabel('Closing Price (USD)')
    plt.show()

    # Plot Quarterly Data (4 boxplots for each quarter)
    plt.figure(figsize=(12, 8))
    data.boxplot(column=close_column, by='Quarter', grid=True)
    plt.title(f'{company_name} - Quarterly Closing Prices (by Quarter)')
    plt.suptitle('')
    plt.xlabel('Quarter')
    plt.ylabel('Closing Price (USD)')
    plt.show()

    return data

# Function to add exogenous variables to the future_dataframe
def add_exogenous_variables(weekly_data: pd.DataFrame):
    """
    Input: future_df (DataFrame) - A DataFrame containing future dates in the 'ds' column.
    Output: future_df (DataFrame) - The same DataFrame with added binary columns for each exogenous event.
    Logic: Adds binary indicators for each specified geopolitical and economic event based on predefined date ranges.
    """
    
    # Define the date ranges for each exogenous event
    covid_start = '2020-01-01'
    covid_end = '2022-12-31'

    geopolitical_tension_start = '2022-02-01'
    geopolitical_tension_end = '2025-12-31'

    trade_sanctions_start = '2018-07-01'
    trade_sanctions_end = '2029-12-31'

    tech_regulation_start = '2020-06-01'
    tech_regulation_end = '2029-12-31'

    new_product_launch_start = '2023-12-01'
    new_product_launch_end = '2025-12-31'

    israel_gaza_conflict_start = '2023-10-07'
    israel_gaza_conflict_end = '2025-12-31'

    # COVID Period
    weekly_data['COVID_Period'] = ((weekly_data.index >= covid_start) & 
                                 (weekly_data.index <= covid_end)).astype(int)
    
    # Geopolitical Tensions (Ukraine Conflict)
    weekly_data['Geopolitical_Tension'] = ((weekly_data.index >= geopolitical_tension_start) & 
                                         (weekly_data.index <= geopolitical_tension_end)).astype(int)
    
    # US-China Trade War
    weekly_data['Trade_Sanctions'] = ((weekly_data.index >= trade_sanctions_start) & 
                                    (weekly_data.index <= trade_sanctions_end)).astype(int)
    
    # Tech Regulation (Export Restrictions)
    weekly_data['Tech_Regulation'] = ((weekly_data.index >= tech_regulation_start) & 
                                    (weekly_data.index <= tech_regulation_end)).astype(int)
    
    # New EUV Machine Launch (TWINSCAN EXE:5000)
    weekly_data['New_Product_Launch'] = ((weekly_data.index >= new_product_launch_start) & 
                                       (weekly_data.index <= new_product_launch_end)).astype(int)
    
    # Israel-Gaza Conflict
    weekly_data['Israel_Gaza_Conflict'] = ((weekly_data.index >= israel_gaza_conflict_start) & 
                                         (weekly_data.index <= israel_gaza_conflict_end)).astype(int)
    

# Function to add exogenous variables to the future_dataframe
def build_exogenous_variables(future_df: pd.DataFrame, list_exogenous: list) -> pd.DataFrame:
    """
    Input: 
        future_df (DataFrame) - A DataFrame containing future dates in the 'ds' column.
        list_exogenous (list) - List of exogenous variables to add.
    Output: 
        future_df (DataFrame) - The same DataFrame with added binary columns for specified exogenous events.
    Logic: 
        Adds binary indicators for each specified exogenous event based on predefined date ranges.
    """
    # Define the date ranges for each exogenous event
    exogenous_events = {
        'COVID_Period': ('2020-01-01', '2022-12-31'),
        'Geopolitical_Tension': ('2022-02-01', '2025-12-31'),
        'Trade_Sanctions': ('2018-07-01', '2029-12-31'),
        'Tech_Regulation': ('2020-06-01', '2029-12-31'),
        'New_Product_Launch': ('2023-12-01', '2025-12-31'),
        'Israel_Gaza_Conflict': ('2023-10-07', '2025-12-31'),
    }

    # Add binary columns for each exogenous variable in the list
    for exogenous in list_exogenous:
        if exogenous in exogenous_events:
            start_date, end_date = exogenous_events[exogenous]
            future_df[exogenous] = ((future_df['ds'] >= start_date) & 
                                    (future_df['ds'] <= end_date)).astype(int)
    
    return future_df