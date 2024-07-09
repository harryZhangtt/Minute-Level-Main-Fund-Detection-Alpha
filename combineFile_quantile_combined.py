import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates


import matplotlib.pyplot as plt
from scipy.stats import zscore

from sklearn.linear_model import LinearRegression

# Define the directory containing the CSV files
# Define the directory containing the CSV files
results_dir = '/Users/zw/Desktop/combined_alpha_results'
merged_csv_dir = '/Users/zw/Desktop/DataBase/Merged_CSVs'  # Replace with the actual path

pnl_dir = os.path.join(results_dir, 'Pnl')
os.makedirs(pnl_dir, exist_ok=True)
max_files = int(1e8)
num_files = 0

# List to store DataFrames
df_list = []

# Function to extract the date from the filename
def extract_date(filename):
    try:
        date_str = filename.split('alpha_')[1].split('.')[0]
        return datetime.strptime(date_str, '%Y%m%d')
    except Exception as e:
        print(f"Error extracting date from filename {filename}: {e}")
        return None

# Get all CSV files in the directory and sort them by date
file_names = sorted(
    [f for f in os.listdir(results_dir) if f.endswith('.csv') and f.startswith('combined_alpha')],
    key=extract_date
)

print("Files to be processed (sorted by date):")

file_dates = []  # List to store dates for debugging

for file_name in file_names:
    if num_files < max_files:
        try:
            file_date = extract_date(file_name)
            file_dates.append(file_date)  # Add file date to the list for debugging
            print(f"{file_name} - Date: {file_date}")
            file_path = os.path.join(results_dir, file_name)
            df = pd.read_csv(file_path)

            # Process the corresponding merged CSV
            merged_file_name = f"merged_{file_date.strftime('%Y%m%d')}.csv"
            merged_file_path = os.path.join(merged_csv_dir, merged_file_name)
            if os.path.exists(merged_file_path):
                merged_df = pd.read_csv(merged_file_path)
                merged_df = merged_df[['SECU_CODE', 'ADJ_CLOSE_PRICE']]
                merged_df = merged_df.sort_values(by='SECU_CODE')

                # Merge with the current df
                df = df.sort_values(by='SECU_CODE')
                df = pd.merge(df, merged_df, on='SECU_CODE', how='left')

            df_list.append(df)
            num_files += 1
            print(f"Processed file: {file_name}")
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    else:
        break

max_date_df_list = max([df['Date'].max() for df in df_list if 'Date' in df.columns])
print(f"Max date in df_list: {max_date_df_list}")

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# Debug print to verify the columns
print("Columns in the combined DataFrame before dropping:")
print(combined_df.columns)

# Rename column if necessary
combined_df.rename(columns={'ADJ_CLOSE_PRICE_y': 'ADJ_CLOSE_PRICE'}, inplace=True)


def calculate_factor_exposure(df, factor_name):
    df = df.copy()
    df['factor_exposure'] = df[factor_name].astype(float)
    return df


def calculate_factor_stability_coefficient(df):
    # Ensure the data is sorted by Date and SECU_CODE
    df = df.sort_values(by=['Date', 'SECU_CODE'])

    # Extract year and month from Date
    df['year_month'] = df['Date'].dt.to_period('M')

    # Initialize a list to store the stability coefficients
    stability_coefficients = []

    # Group by year_month
    grouped = df.groupby('year_month')

    # Get the unique periods
    periods = df['year_month'].unique()

    # Loop through consecutive periods to calculate the cross-sectional correlation
    for i in range(1, len(periods)):
        prev_period = periods[i - 1]
        current_period = periods[i]

        # Get the factor exposures for the previous and current periods
        prev_exposures = grouped.get_group(prev_period)[['SECU_CODE', 'factor_exposure']].set_index('SECU_CODE')
        current_exposures = grouped.get_group(current_period)[['SECU_CODE', 'factor_exposure']].set_index(
            'SECU_CODE')

        # Join the exposures on SECU_CODE
        merged_exposures = prev_exposures.join(current_exposures, lsuffix='_prev', rsuffix='_curr', how='inner')

        # Ensure both DataFrames have the same length and there are enough data points
        min_length = min(len(prev_exposures), len(current_exposures))
        if min_length > 1:
            prev_exposures = prev_exposures.iloc[:min_length]
            current_exposures = current_exposures.iloc[:min_length]

            # Calculate the cross-sectional correlation
            correlation = prev_exposures['factor_exposure'].corr(current_exposures['factor_exposure'])

            # Store the stability coefficient
            stability_coefficients.append({
                'year_month': current_period,
                'factor_stability_coefficient': correlation
            })

    # Convert to DataFrame
    stability_df = pd.DataFrame(stability_coefficients)
    # Initialize the factor_stability_coefficient column in df
    df['factor_stability_coefficient'] = np.nan

    # Assign stability coefficients to the original dataframe using .loc
    for index, row in stability_df.iterrows():
        df.loc[df['year_month'] == row['year_month'], 'factor_stability_coefficient'] = row[
            'factor_stability_coefficient']

    return df
# Perform the backtest on the combined DataFrame
def vector_processing(df):
    df = df.sort_values(by=['Date', 'SECU_CODE'], ascending=True)
    # Calculate 'vector_1' as the rolling mean of 'main_fund_alpha' grouped by 'SECU_CODE'
    df['vector_1'] = df.groupby('SECU_CODE')['main_fund_alpha'].rolling(window=20, min_periods=1).mean().reset_index(
        level=0, drop=True)

    # Calculate 'vector_2' as the rolling mean of 'raw_volume_alpha' grouped by 'SECU_CODE'
    df['vector_2'] = df.groupby('SECU_CODE')['raw_volume_alpha'].rolling(window=20, min_periods=1).mean().reset_index(
        level=0, drop=True)

    df['vector_3'] = df.groupby('SECU_CODE')['main_fund_volume_alpha'].rolling(window=20,
                                                                               min_periods=1).mean().reset_index(
        level=0, drop=True)

    def _calculate_residuals(group, vector_column, response_column):
        X = group[vector_column].values.reshape(-1, 1)
        y = group[response_column].values.reshape(-1, 1)

        valid_indices = ~np.isnan(X).flatten() & ~np.isnan(y).flatten()
        X = X[valid_indices]
        y = y[valid_indices]

        if len(X) == 0 or len(y) == 0:
            return pd.Series([np.nan] * len(group), index=group.index)

        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        residuals = y - y_pred

        alpha = pd.Series(np.nan, index=group.index)
        alpha.iloc[valid_indices] = residuals.flatten()
        return alpha

    # Calculate 'industry_neutral_vector_1' by subtracting the mean of 'vector_1' within each industry and date group
    df['industry_neutral_vector_1'] = df['vector_1'] - df.groupby(['industry', 'Date'])['vector_1'].transform('mean')

    # Calculate 'industry_neutral_vector_2' by subtracting the mean of 'vector_2' within each industry and date group
    df['industry_neutral_vector_2'] = df['vector_2'] - df.groupby(['industry', 'Date'])['vector_2'].transform('mean')

    # Calculate 'industry_neutral_vector_3' by subtracting the mean of 'vector_3' within each industry and date group
    df['industry_neutral_vector_3'] = df['vector_3'] - df.groupby(['industry', 'Date'])['vector_3'].transform('mean')

    # Apply the residuals calculation function to each group defined by 'SECU_CODE'
    df['industry_size_neutral_main_fund_alpha'] = df.groupby('Date', group_keys=False).apply(
        lambda group: _calculate_residuals(group, 'mad_log_size', 'industry_neutral_vector_1'))

    df['industry_size_neutral_raw_volume_alpha'] = df.groupby('Date', group_keys=False).apply(
        lambda group: _calculate_residuals(group, 'mad_log_size', 'industry_neutral_vector_2'))

    df['industry_size_neutral_main_fund_volume_alpha'] = df.groupby('Date', group_keys=False).apply(
        lambda group: _calculate_residuals(group, 'mad_log_size', 'industry_neutral_vector_3'))

    # Normalize the industry size neutral alphas
    df['normalized_industry_size_neutral_main_fund_alpha'] = df.groupby('Date')[
        'industry_size_neutral_main_fund_alpha'].transform(lambda x: (x - x.mean()) / x.std())
    df['normalized_industry_size_neutral_raw_volume_alpha'] = df.groupby('Date')[
        'industry_size_neutral_raw_volume_alpha'].transform(lambda x: (x - x.mean()) / x.std())
    df['normalized_industry_size_neutral_main_fund_volume_alpha'] = df.groupby('Date')[
        'industry_size_neutral_main_fund_volume_alpha'].transform(lambda x: (x - x.mean()) / x.std())

    # Combine the alphas with the z-score normalization applied on a daily basis
    df['combined_alpha'] = -1 * df['normalized_industry_size_neutral_main_fund_alpha'] + df[
        'normalized_industry_size_neutral_raw_volume_alpha']
    df['normalized_combined_alpha'] = df.groupby('Date')['combined_alpha'].transform(lambda x: (x - x.mean()) / x.std())
    return df


def simple_backtest(df, initial_capital=1e8):
    df = df.sort_values(by=['Date','SECU_CODE'],ascending=True)
    if 'ADJ_CLOSE_PRICE' not in df.columns:
        print("ADJ_CLOSE_PRICE column is missing.")
        return None, None

    # Ensure there are no zero prices to avoid division by zero
    df['ADJ_CLOSE_PRICE'] = df['ADJ_CLOSE_PRICE'].replace(0, np.nan)
    df['ADJ_CLOSE_PRICE'] = df['ADJ_CLOSE_PRICE'].ffill()

    # Append the vector to the DataFrame to align by date and stock code, assuming vector is correctly indexed
    df=vector_processing(df)

    vectors = [
        'normalized_industry_size_neutral_main_fund_alpha',
        'normalized_industry_size_neutral_raw_volume_alpha',
        'normalized_industry_size_neutral_main_fund_volume_alpha',
        'normalized_combined_alpha'
    ]

    for vector in vectors:
        if vector== 'normalized_industry_size_neutral_main_fund_alpha':
            df['vector']= -1*df[vector]
        else:
            df['vector']= df[vector]
        df['vector'] = df.groupby('SECU_CODE')['vector'].shift(1)
        df['vector'] = df['vector'].fillna(0)

        # Ensure the vector column is numeric
        df['vector'] = df['vector'].astype(float)


        df['vector'] = df['vector'].fillna(0)

        # Ensure the vector column is numeric
        df['vector'] = df['vector'].astype(float)

        def weight_assignment(df):
            df = df.sort_values(by='Date')

            # Function to add small noise to avoid duplicates
            def add_noise(series):
                return series + np.random.normal(0, 1e-6, len(series))

            # Calculate quantiles for vector within each group
            def quantile_transform(group):
                try:
                    group['quantile'] = pd.qcut(group['vector'], q=5, labels=False)
                except ValueError:
                    group['quantile'] = pd.qcut(add_noise(group['vector']), q=5, labels=False)
                return group

            df = df.groupby('Date', group_keys=False).apply(quantile_transform)

            # Define masks for long and short investments based on the quantiles
            df['long_weight'] = 0.0
            df['short_weight'] = 0.0

            top_20_mask = df['quantile'] == 4  # Top 20% (highest quantile)
            bottom_20_mask = df['quantile'] == 0  # Bottom 20% (lowest quantile)

            df.loc[top_20_mask, 'long_weight'] = abs(df['vector']) / df[top_20_mask].groupby('Date')['vector'].transform(
                'sum')
            df.loc[bottom_20_mask, 'short_weight'] = abs(df['vector']) / df[bottom_20_mask].groupby('Date')[
                'vector'].transform('sum')

            df['weight'] = 0.0
            df.loc[top_20_mask, 'weight'] = df['long_weight']
            df.loc[bottom_20_mask, 'weight'] = df['short_weight']

            return df

        df = weight_assignment(df)

        # Allocate capital based on weights
        df['long_capital_allocation'] = initial_capital * df['long_weight']
        df['short_capital_allocation'] = initial_capital * df['short_weight']

        # Calculate investment amount in shares for long and short positions
        # round to the closest multiple of 100 smaller than the original number
        df['long_investments'] = ((df['long_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100
        df['short_investments'] = ((df['short_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100

        # Assign investments based on the vector value
        df['investment'] = 0  # Initialize investment column with zeros

        # Assign long investments to stocks with positive vector
        df.loc[df['weight'] >= 0, 'investment'] = df['long_investments']

        # Assign short investments to stocks with negative vector
        df.loc[df['weight'] < 0, 'investment'] = df['short_investments']

        print(df['investment'])

        # Calculate the next-day price change
        df['next_day_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].diff()

        df['next_day_return'] = df['next_day_return'].fillna(0)  # Fill NaNs that result from diff and shift

        # Shift investments to get the previous day's investments
        df['previous_investment'] = df.groupby('SECU_CODE')['investment'].shift(1)

        # Calculate investment changes
        df['investment_change'] = (df['investment'] - df['previous_investment']).fillna(0)
        df['abs_investment_change'] = abs(df['investment_change'])

        # Calculate hold_pnl based on the condition
        condition = df['previous_investment'] * df['investment'] > 0
        # df['hold_pnl'] = np.where(condition, df['previous_investment'] * df['next_day_return'], 0)
        # df['adj_factor'] = np.where(df['cp'] != 0, df['ADJ_CLOSE_PRICE'] / df['cp'], 0)
        # df['ADJ_VWAP'] = df['adj_factor'] * df['VWAP']
        # df['trade_return'] = np.where(df['cp'] != 0, df['ADJ_CLOSE_PRICE'] - df['ADJ_VWAP'], 0)
        #
        # df['trade_pnl'] = df['investment_change'] * df['trade_return']
        # df['pnl'] = df['hold_pnl'] + df['trade_pnl']
        df['pnl']= df['next_day_return']*df['investment']
        df['pnl'] = df['pnl'].fillna(0)
        df['long_pnl'] = np.where(df['weight'] > 0, df['pnl'], 0)
        df['short_pnl'] = np.where(df['weight'] < 0, df['pnl'], 0)

        df['Date'] = pd.to_datetime(df['Date'])

        # Calculate TVR Shares and TVR Values
        df['tvr_shares'] = df['abs_investment_change']
        df['tvr_values'] = df['abs_investment_change'] * df['ADJ_CLOSE_PRICE']
        df['tvr_shares'] = df['tvr_shares'].fillna(0)
        df['tvr_values'] = df['tvr_values'].fillna(0)
        df['TOTALVALUE']=df['TRADABLE_SHARES']*df['ADJ_CLOSE_PRICE']
        df = calculate_factor_exposure(df, factor_name=vector)
        df = calculate_factor_stability_coefficient(df)
        df.rename(columns={'factor_stability_coefficient_y': 'factor_stability_coefficient'}, inplace=True)
        print(df['factor_stability_coefficient'])

        aggregated = df.groupby('Date').agg(
            pnl=('pnl', 'sum'),
            long_pnl=('long_pnl', 'sum'),
            short_pnl=('short_pnl', 'sum'),
            long_size=('investment', lambda x: (
                        x[(df.loc[x.index, 'vector'] >= 0) & (df.loc[x.index, 'quantile'] == 4)] * df.loc[
                    x.index, 'ADJ_CLOSE_PRICE']).sum()),
            short_size=('investment', lambda x: (
                        -x[(df.loc[x.index, 'vector'] < 0) & (df.loc[x.index, 'quantile'] == 0)] * df.loc[
                    x.index, 'ADJ_CLOSE_PRICE']).sum()),
            total_size=('investment', lambda x: (x[(df.loc[x.index, 'vector'] >= 0) & (df.loc[x.index, 'quantile'] == 4)] *
                                                 df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum() +
                                                (-x[(df.loc[x.index, 'vector'] < 0) & (df.loc[x.index, 'quantile'] == 0)] *
                                                 df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum()),
            tvrshares=('tvr_shares', 'sum'),
            tvrvalues=('tvr_values', 'sum'),
            long_count=('vector', lambda x: (
                (x[df.loc[x.index, 'quantile'] == 4]).ge(x.shift(1)[df.loc[x.index, 'quantile'] == 4])).sum()),
            short_count=('vector', lambda x: (
                (x[df.loc[x.index, 'quantile'] == 0]).lt(x.shift(1)[df.loc[x.index, 'quantile'] == 0])).sum())
        ).reset_index()

        aggregated['cum_pnl'] = aggregated['pnl'].cumsum() / (2 * initial_capital)
        aggregated['cum_long_pnl'] = aggregated['long_pnl'].cumsum() / (2 * initial_capital)
        aggregated['cum_short_pnl'] = aggregated['short_pnl'].cumsum() / (2 * initial_capital)

        # Extract year from Date
        aggregated['year'] = aggregated['Date'].dt.year

        # Calculate annualized return for each year
        annual_returns = aggregated.groupby('year')['pnl'].sum().reset_index()
        annual_returns.columns = ['year', 'annualized_return']
        annual_returns['annualized_return'] = annual_returns['annualized_return'] / (2 * initial_capital)

        # Merge annualized return back to aggregated DataFrame
        aggregated = pd.merge(aggregated, annual_returns, on='year', how='left')
        # Calculate Sharpe Ratio
        daily_returns = (aggregated['pnl'] / 2 * initial_capital).fillna(0)
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        aggregated['sharpe_ratio'] = sharpe_ratio

        def calculate_max_drawdown(df):
            df = df.sort_values(by='Date')  # Ensure data is sorted by date
            max_drawdown = 0
            for i in range(1, len(df)):
                drawdown = aggregated.loc[i, 'pnl'] / initial_capital
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
                df.loc[i, 'mdd'] = max_drawdown

            return df

        aggregated = calculate_max_drawdown(aggregated)

        df['stocks_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].pct_change()
        df['stocks_return'] = df['stocks_return'].fillna(0)

        # Calculate Information Coefficient (IC)
        aggregated['IC'] = aggregated['Date'].apply(
            lambda day: df.loc[df['Date'] == day, 'vector'].corr(df.loc[df['Date'] == day, 'stocks_return'])
        )
        aggregated['cum_IC'] = aggregated['IC'].cumsum()
        overall_pnl = df['pnl'].sum()

        def plot_combined_graphs(aggregated, df, initial_principal, vector):
            # Ensure TRADINGDAY_x is treated as datetime
            aggregated['Date'] = pd.to_datetime(aggregated['Date'], format='%Y%m%d')
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df['pct_return'] = np.log(df['ADJ_CLOSE_PRICE'] / df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(1))
            cumulative_avg_return = df.groupby('Date')['pct_return'].mean().cumsum()

            # Calculate TVR ratio
            aggregated['tvr_ratio'] = aggregated['tvrvalues'] / initial_principal

            # Calculate excess returns
            aggregated[f'{vector}_excess_pnl'] = aggregated['cum_pnl'] - cumulative_avg_return.reindex(aggregated['Date']).values
            aggregated[f'{vector}_excess_long_pnl'] = aggregated[
                                                               'cum_long_pnl'] - cumulative_avg_return.reindex(aggregated['Date']).values
            aggregated[f'{vector}_excess_short_pnl'] = aggregated[
                                                                'cum_short_pnl'] - cumulative_avg_return.reindex(aggregated['Date']).values

            fig, axs = plt.subplots(3, 1, figsize=(10, 8))

            # Plot cumulative PnL
            axs[0].plot(aggregated['Date'], aggregated['cum_pnl'], label='Cumulative PnL')
            axs[0].plot(aggregated['Date'], aggregated['cum_long_pnl'], label='Cumulative Long PnL')
            axs[0].plot(aggregated['Date'], aggregated['cum_short_pnl'], label='Cumulative Short PnL')
            axs[0].plot(cumulative_avg_return.index, cumulative_avg_return.values,
                        label='Cumulative Average Return')
            axs[0].xaxis.set_major_locator(mdates.YearLocator())
            axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[0].set_title('Cumulative PnL, Long PnL, Short PnL, and Cumulative Average Return', fontsize='small')
            axs[0].set_xlabel('Trading Day', fontsize='small')
            axs[0].set_ylabel('Cumulative Return', fontsize='small')
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            axs[0].grid(True)

            # Plot histogram of TVR ratio
            axs[1].hist(aggregated['tvr_ratio'], bins=30, color='blue', edgecolor='black', alpha=0.7)
            axs[1].set_title('Distribution of TVR Ratio', fontsize='small')
            axs[1].set_xlabel('TVR Ratio', fontsize='small')
            axs[1].set_ylabel('Frequency', fontsize='small')
            axs[1].grid(True)

            # Plot annualized return
            axs[2].plot(aggregated['Date'], aggregated['annualized_return'], label='Annualized Return')
            axs[2].xaxis.set_major_locator(mdates.YearLocator())
            axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[2].set_title('Annualized Return Over Time', fontsize='small')
            axs[2].set_xlabel('Trading Day', fontsize='small')
            axs[2].set_ylabel('Annualized Return', fontsize='small')
            axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            axs[2].grid(True)

            fig, axs = plt.subplots(2, 1, figsize=(14, 8))
            # Plot excess returns
            axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_pnl'], label='Excess PnL')
            axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_long_pnl'],
                        label='Excess Long PnL')
            axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_short_pnl'],
                        label='Excess Short PnL')
            axs[0].xaxis.set_major_locator(mdates.YearLocator())
            axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[0].set_title('Excess Returns (Overall, Long, Short)', fontsize='small')
            axs[0].set_xlabel('Trading Day', fontsize='small')
            axs[0].set_ylabel('Excess Return', fontsize='small')
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            axs[0].grid(True)
            # Plot excess returns
            axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_pnl'], label='Excess PnL')
            axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_long_pnl'], label='Excess Long PnL')
            axs[0].plot(aggregated['Date'], aggregated[f'{vector}_excess_short_pnl'], label='Excess Short PnL')

            # Set major locator and formatter for x-axis
            axs[0].xaxis.set_major_locator(mdates.YearLocator())
            axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # Set title, labels, legend, and grid for the first plot
            axs[0].set_title('Excess Returns (Overall, Long, Short)', fontsize='small')
            axs[0].set_xlabel('Trading Day', fontsize='small')
            axs[0].set_ylabel('Excess Return', fontsize='small')
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            axs[0].grid(True)

            # Plot cumulative IC
            axs[1].plot(aggregated['Date'], aggregated['cum_IC'], label='Cumulative IC')

            # Set major locator and formatter for x-axis
            axs[1].xaxis.set_major_locator(mdates.YearLocator())
            axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # Set title, labels, legend, and grid for the second plot
            axs[1].set_title('Cumulative IC', fontsize='small')
            axs[1].set_xlabel('Trading Day', fontsize='small')
            axs[1].set_ylabel('Cumulative IC', fontsize='small')
            axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            axs[1].grid(True)


            plt.tight_layout()
            plt.show()

        plot_combined_graphs(aggregated, df, initial_capital, vector)

        def grouping_analysis(df):
            # Sort the dataframe by vector and trading day for proper quantile grouping
            df = df.sort_values(by=['Date', 'vector'])

            # Initialize a list to store average returns per day
            average_returns_per_day_list = []
            average_size_per_day_list = []


            # Loop over each trading day to calculate daily average returns for each vector group
            for trading_day in df['Date'].unique():
                daily_df = df[df['Date'] == trading_day].copy()
                daily_df['vector_group'] = pd.qcut(daily_df['vector'], q=50, labels=False, duplicates='drop')
                daily_average_returns = daily_df.groupby('vector_group')['pct_return'].mean().reset_index()
                daily_average_size = daily_df.groupby('vector_group')['TOTALVALUE'].mean().reset_index()
                daily_average_returns['Date'] = trading_day
                daily_average_size['Date'] = trading_day
                average_returns_per_day_list.append(daily_average_returns)
                average_size_per_day_list.append(daily_average_size)

            # Concatenate the daily average returns into a single dataframe
            average_returns_per_day = pd.concat(average_returns_per_day_list)
            average_size_per_day = pd.concat(average_size_per_day_list)
            average_returns_per_day.set_index('Date', inplace=True)
            average_size_per_day.set_index('Date', inplace=True)

            # Calculate the overall average return and size for each vector group across all days
            average_returns = average_returns_per_day.groupby('vector_group')['pct_return'].mean().reset_index()
            average_size = average_size_per_day.groupby('vector_group')['TOTALVALUE'].mean().reset_index()

            # Rename columns for clarity
            average_returns.columns = ['vector_group', 'average_return']
            average_returns = average_returns.sort_values(by='vector_group', ascending=True)
            average_size.columns = ['vector_group', 'average_size']
            average_size = average_size.sort_values(by='vector_group', ascending=True)

            # Create subplots
            fig, axs = plt.subplots(2, 1, figsize=(14, 10))

            # Plotting average returns
            axs[0].bar(average_returns['vector_group'], average_returns['average_return'], color='b', alpha=0.6)
            axs[0].set_xlabel('Vector Group', fontsize='small')
            axs[0].set_ylabel('Average Return', fontsize='small')
            axs[0].set_title('Average Return by Vector Group', fontsize='small')
            axs[0].grid(True)

            # Plotting average sizes
            axs[1].bar(average_size['vector_group'], average_size['average_size'], color='r', alpha=0.6)
            axs[1].set_xlabel('Vector Group', fontsize='small')
            axs[1].set_ylabel('Average Size', fontsize='small')
            axs[1].set_title('Average Size by Vector Group', fontsize='small')
            axs[1].grid(True)

            plt.tight_layout()  # Add more space between plots
            plt.show()
        grouping_analysis(df)

        print(df['pnl'])
        print(overall_pnl)
        output_file = os.path.join(pnl_dir, f'{vector}_combined_aggregated.csv')
        aggregated.to_csv(output_file,index=False)
        print(f'results for {vector} saved to {output_file}')

simple_backtest(combined_df)


