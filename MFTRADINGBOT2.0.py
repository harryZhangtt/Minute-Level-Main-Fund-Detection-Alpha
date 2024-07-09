import os
import numpy as np
import pandas as pd
import gc
from collections import deque

from sklearn.linear_model import LinearRegression
from scipy.stats import zscore


class DataCombiner:
    def __init__(self, base_paths, sk_base_path,mv_liq_free_path,cp_base_path,industry_file_path, max_length=7):
        self.max_length= max_length
        self.base_paths = base_paths
        self.sk_base_path = sk_base_path
        self.mv_liq_free_path = mv_liq_free_path
        self.cp_base_path= cp_base_path
        self.industry_data = pd.read_excel(industry_file_path, engine='openpyxl')
        self.queue = deque(maxlen=max_length)

        self.pv_df=pd.DataFrame
        self.daily_data= pd.DataFrame
        self.latest_data=pd.DataFrame
        self.all_results=[]

    
    def read_and_concatenate(self, cp_subdir, cp_prefix, v_subdir, v_prefix, max_files=None):
        files_processed = 0
        for base_path in self.base_paths:
            cp_data_dir = os.path.join(base_path, cp_subdir)
            v_data_dir = os.path.join(base_path, v_subdir)

            if os.path.exists(cp_data_dir) and os.path.exists(v_data_dir):
                cp_files = sorted([f for f in os.listdir(cp_data_dir) if
                                   os.path.isfile(os.path.join(cp_data_dir, f)) and f.startswith(
                                       cp_prefix) and f.endswith('.txt')])
                v_files = sorted([f for f in os.listdir(v_data_dir) if
                                  os.path.isfile(os.path.join(v_data_dir, f)) and f.startswith(v_prefix) and f.endswith(
                                      '.txt')])

                for cp_filename, v_filename in zip(cp_files, v_files):
                    if max_files and files_processed >= max_files:
                        break

                    print(f"Processing files: {cp_filename} and {v_filename} from {base_path}")
                    cp_file_path = os.path.join(cp_data_dir, cp_filename)
                    v_file_path = os.path.join(v_data_dir, v_filename)

                    try:
                        cp_data = pd.read_csv(cp_file_path, delimiter='\t', encoding='utf-8')
                        v_data = pd.read_csv(v_file_path, delimiter='\t', encoding='utf-8')
                        """
                        1. read and rename the close price and volume txt"""
                        # Extract date from filename
                        date_str = cp_filename.split('_')[1].split('.')[0]
                        cp_data.insert(0, 'Date', date_str)
                        v_data.insert(0, 'Date', date_str)

                        # Create and rename columns
                        cp_data.columns = ['Date'] + [f"{i}_cp" for i in range(1, cp_data.shape[1])]
                        v_data.columns = ['Date'] + [f"{i}_v" for i in range(1, v_data.shape[1])]

                        # Process SK Files
                        """
                        2. add secu code to the csv
                        """
                        cp_data = self.process_sk_files(cp_data)
                        v_data = self.process_sk_files(v_data)

                        # Ensure the lengths match before merging
                        min_length = min(len(cp_data), len(v_data))
                        cp_data = cp_data.iloc[:min_length].copy()
                        v_data = v_data.iloc[:min_length].copy()

                        # Convert expanded dataframes to long format
                        num_cp_columns = cp_data.shape[1] - 1
                        num_v_columns = v_data.shape[1] - 1
                        """
                        3.convert to long format"""
                        cp_long = self.convert_to_long_format(cp_data, num_cp_columns, 'cp')
                        v_long = self.convert_to_long_format(v_data, num_v_columns, 'v')

                        # Take the intersection of Date and SECU_CODE
                        intersected_data = pd.merge(cp_long[['Date', 'SECU_CODE', 'minute']],
                                                    v_long[['Date', 'SECU_CODE', 'minute']],
                                                    on=['Date', 'SECU_CODE', 'minute'], how='inner')

                        # Merge the price and volume data
                        """
                        4. merge the price and volume csv"""
                        merged_df = pd.merge(intersected_data, cp_long, on=['Date', 'SECU_CODE', 'minute'], how='left')
                        merged_df = pd.merge(merged_df, v_long, on=['Date', 'SECU_CODE', 'minute'], how='left')

                        print(f"Merged data for date {date_str}:\n{merged_df}")  # Debug print

                        # Enqueue the long format data for processing with SK files
                        self.queue.append(merged_df)

                        files_processed += 1

                        if files_processed >= self.max_length - 1:
                            # Process the queue
                            self.pv_df = self.queue_to_dataframe()
                            self.calculate_volume_critical()
                            self.merge_total_share()

                            # Calculate combined alphas
                            self.calculate_combined_alpha()
                            self.all_results.append(self.latest_data.copy())
                            print(self.all_results)

                            if files_processed >= self.max_length:
                                self.queue.popleft()

                            gc.collect()

                    except pd.errors.ParserError as e:
                        print(
                            f"Parser error processing files: {cp_file_path} and {v_file_path}. Error: {e}. Trying with different encoding.")
                        continue
                    except Exception as e:
                        print(f"Error processing data for {cp_file_path} and {v_file_path}: {e}")
                        continue

    def queue_to_dataframe(self):
        """
        Convert the queue of dataframes to a single concatenated dataframe.
        """
        if not self.queue:
            return pd.DataFrame()  # Return an empty dataframe if the queue is empty

        return pd.concat(list(self.queue), ignore_index=True)


    def expand_and_reformat_dataframe(self, df):
        expanded_data = []

        for _, row in df.iterrows():
            new_row = []
            for cell in row:
                values = str(cell).split()
                if len(values) > 1:
                    if new_row:
                        expanded_data.append(new_row)
                    for value in values:
                        expanded_data.append([value])
                    new_row = []
                else:
                    new_row.extend(values)

            if new_row:
                expanded_data.append(new_row)

        max_length = max(len(row) for row in expanded_data)
        for row in expanded_data:
            while len(row) < max_length:
                row.append(None)

        new_df = pd.DataFrame(expanded_data)
        return new_df

    def process_sk_files(self, concatenated_data):
        sk_files = os.listdir(self.sk_base_path)

        # Iterate over unique dates in the concatenated data
        for date in concatenated_data['Date'].unique():
            print(f"Processing SK file for date: {date}")
            sk_filename = f"Sk_{date}.txt"
            sk_file_path = os.path.join(self.sk_base_path, sk_filename)

            if sk_filename in sk_files:
                sk_data = pd.read_csv(sk_file_path, delimiter='\t', encoding='utf-8', engine='python')
                if not sk_data.empty:
                    secu_codes = sk_data.iloc[:, 0].tolist()  # Extract all ticker names from the first column

                    # Assign SECU_CODE to the corresponding date rows
                    date_rows = concatenated_data[concatenated_data['Date'] == date]
                    min_length = min(len(date_rows), len(secu_codes))

                    concatenated_data.loc[date_rows.index[:min_length], 'SECU_CODE'] = secu_codes[:min_length]

        return concatenated_data

    def convert_to_long_format(self, df, num_columns, suffix):
        # Identify columns
        columns = [f'{i}_{suffix}' for i in range(2, num_columns)]

        # Melt the dataframe to long format
        melted_df = df.melt(id_vars=['Date','SECU_CODE'], value_vars=columns, var_name='minute', value_name=suffix)

        # Extract minute from the column names
        melted_df['minute'] = melted_df['minute'].str.extract('(\d+)').astype(int)

        return melted_df


    def merge_total_share(self):
        # Initialize a list to store tradable shares, industry data, and CP data for all dates
        merged_data_list = []

        # Get a list of all MvLiqFree and CP files and sort them in chronological order
        mv_files = sorted([os.path.join(self.mv_liq_free_path, f) for f in os.listdir(self.mv_liq_free_path)
                           if f.startswith("MvLiqFree_") and f.endswith(".txt")])
        cp_files = sorted([os.path.join(self.cp_base_path, f) for f in os.listdir(self.cp_base_path)
                           if f.startswith("CP_") and f.endswith(".txt")])

        # Filter files to start from 20210104 and limit the number of files to 10
        mv_files = [f for f in mv_files if int(os.path.basename(f).split('_')[1].split('.')[0]) >= 20210104]
        cp_files = [f for f in cp_files if int(os.path.basename(f).split('_')[1].split('.')[0]) >= 20210104]

        # Create a copy of self.pv_df to work on
        pv_df = self.pv_df.copy()

        # Read and process each MvLiqFree and corresponding CP file
        for mv_file, cp_file in zip(mv_files, cp_files):
            # Read MvLiqFree data
            liq_free_data = pd.read_csv(mv_file, delimiter='\t', header=None, encoding='utf-8')

            # Read CP data
            cp_data = pd.read_csv(cp_file, delimiter='\t', header=None, encoding='utf-8')

            # Ensure the 'Date' column is in datetime format
            date_str = os.path.basename(mv_file).split('_')[1].split('.')[0]
            date = pd.to_datetime(date_str, format='%Y%m%d')

            # Get the unique SECU_CODEs for the given date
            secu_codes = pv_df.loc[pv_df['Date'] == date, 'SECU_CODE'].unique()

            num_securities = min(len(secu_codes), len(liq_free_data), len(cp_data))

            liq_free_data = liq_free_data.iloc[:num_securities]
            cp_data = cp_data.iloc[:num_securities]
            secu_codes = secu_codes[:num_securities]

            # Assume the industry data is aligned and extract the corresponding industries
            industry_info = self.industry_data['industry'].iloc[:num_securities].values

            liq_free_data['Date'] = date
            liq_free_data['SECU_CODE'] = secu_codes
            liq_free_data['industry'] = industry_info
            liq_free_data.columns = ['TRADABLE_SHARES', 'Date', 'SECU_CODE', 'industry']

            cp_data['Date'] = date
            cp_data['SECU_CODE'] = secu_codes
            cp_data.columns = ['ADJ_CLOSE_PRICE', 'Date', 'SECU_CODE']

            # Merge the data for the given date
            merged_data = liq_free_data[['Date', 'SECU_CODE', 'TRADABLE_SHARES', 'industry']].copy()
            merged_data['ADJ_CLOSE_PRICE'] = cp_data['ADJ_CLOSE_PRICE'].values

            merged_data_list.append(merged_data)

        # Concatenate all the merged data into a single DataFrame
        merged_data_combined = pd.concat(merged_data_list, ignore_index=True)

        # Set the index to ['Date', 'SECU_CODE'] for both DataFrames for alignment
        pv_df.set_index(['Date', 'SECU_CODE'], inplace=True)
        merged_data_combined.set_index(['Date', 'SECU_CODE'], inplace=True)

        # Map the TRADABLE_SHARES, industry, and ADJ_CLOSE_PRICE from merged_data_combined to pv_df
        pv_df = pv_df.join(merged_data_combined[['TRADABLE_SHARES', 'industry', 'ADJ_CLOSE_PRICE']], how='left')

        # Reset the index of pv_df
        pv_df.reset_index(inplace=True)

        # Assign the modified pv_df back to self.pv_df
        self.pv_df = pv_df

        # Print the updated DataFrame
        print("Updated self.pv_df with TRADABLE_SHARES, industry, and ADJ_CLOSE_PRICE:")
        print(self.pv_df[['Date', 'SECU_CODE', 'TRADABLE_SHARES', 'industry', 'ADJ_CLOSE_PRICE']])



    def calculate_volume_critical(self):
        df = self.pv_df.copy()
        df = df.sort_values(by=['Date', 'SECU_CODE'])

        # Convert Date to datetime for rolling function
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Shift the volume by one to avoid future data leakage
        df['VOLUME_SHIFTED'] = df.groupby('SECU_CODE')['v'].shift(1)

        # Calculate the rolling 90th percentile of shifted volume for the past 5 days within each ticker
        df['Vcritical'] = df.groupby('SECU_CODE')['VOLUME_SHIFTED'].transform(
            lambda x: x.rolling(window=5 * 240, min_periods=5 * 240).quantile(0.90))

        # Save the updated data
        self.pv_df = df

        return self.pv_df[['SECU_CODE', 'Date', 'Vcritical']]

    def calculate_combined_alpha(self):
        self._prepare_data()

        self._calculate_main_fund_weighted()
        self._calculate_main_fund_supres()

        self._update_daily_data()
        print(1)
        self._calculate_normalized_alpha('main_fund_alpha', 'main_fund_rank', 'normalized_main_fund_alpha')
        print(2)
        self._calculate_normalized_alpha('raw_volume_alpha', 'supres_rank', 'normalized_volume_alpha')
        print(3)
        self._calculate_industry_neutral_alpha('normalized_main_fund_alpha', 'industry_neutral_main_fund_alpha')
        print(4)
        self._calculate_industry_neutral_alpha('normalized_volume_alpha', 'industry_neutral_volume_alpha')
        print(5)

        self._calculate_industry_size_neutral_alpha('log_size', 'industry_neutral_main_fund_alpha',
                                                    'industry_size_neutral_main_fund_alpha')
        print(6)
        self._calculate_industry_size_neutral_alpha('log_size', 'industry_neutral_volume_alpha',
                                                    'industry_size_neutral_volume_alpha')
        print(7)

        self._calculate_combined_alpha()
        print(8)

        print(self.latest_data)

    def _prepare_data(self):
        self.pv_df.sort_values(by=['Date', 'SECU_CODE'], inplace=True)
        self.pv_df['Date'] = pd.to_datetime(self.pv_df['Date'], errors='coerce')

    def _calculate_main_fund_weighted(self):
        df_filtered = self._filter_data_by_volume_and_time(self.pv_df)
        df_grouped = self._calculate_weighted_price(df_filtered)
        self.pv_df = self._merge_with_main_fund_weighted(self.pv_df, df_grouped)
        self._calculate_vwap()
        self.pv_df['main_fund_alpha'] = np.where(self.pv_df['VWAP'] != 0,
                                                 self.pv_df['MAIN_FUND_WEIGHTED'] / self.pv_df['VWAP'] - 1, 0)

    def _calculate_main_fund_supres(self):
        df = self.pv_df.copy()
        df['daily_average_cp'] = df.groupby(['Date', 'SECU_CODE'])['cp'].transform('mean')

        # Calculate support volume
        df['support_volume'] = df['v'] * (df['cp'] <= df['daily_average_cp'])
        df['support_volume'] = df.groupby(['Date', 'SECU_CODE'])['support_volume'].transform('sum')

        # Calculate resistance volume
        df['res_volume'] = df['v'] * (df['cp'] > df['daily_average_cp'])
        df['res_volume'] = df.groupby(['Date', 'SECU_CODE'])['res_volume'].transform('sum')

        # Calculate raw volume alpha
        df['raw_volume_alpha'] = np.where(df['TRADABLE_SHARES'] != 0,
                                          (df['support_volume'] - df['res_volume']) / df['TRADABLE_SHARES'], 0)

        self.pv_df = df

    def _filter_data_by_volume_and_time(self, df):
        time_condition = (df['minute'] > 10) & (df['minute'] < 237)
        return df[(df['v'] >= df['Vcritical']) & time_condition].copy()

    def _calculate_weighted_price(self, df):
        df['WEIGHTED_PRICE'] = df['cp'] * df['v']
        return df.groupby(['SECU_CODE', 'Date']).agg({'WEIGHTED_PRICE': 'sum', 'v': 'sum'}).reset_index()

    def _merge_with_main_fund_weighted(self, df, df_grouped):
        df_grouped['MAIN_FUND_WEIGHTED'] = df_grouped['WEIGHTED_PRICE'] / df_grouped['v']
        return df.merge(df_grouped[['SECU_CODE', 'Date', 'MAIN_FUND_WEIGHTED']], on=['SECU_CODE', 'Date'], how='left')

    def _calculate_vwap(self):
        self.pv_df['WEIGHTED_PRICE'] = self.pv_df['cp'] * self.pv_df['v']
        daily_aggregates = self.pv_df.groupby(['Date', 'SECU_CODE']).agg(
            {'WEIGHTED_PRICE': 'sum', 'v': 'sum'}).reset_index()
        daily_aggregates['DAILY_WEIGHTED'] = daily_aggregates['WEIGHTED_PRICE'] / daily_aggregates['v']
        self.pv_df = self.pv_df.merge(daily_aggregates[['Date', 'SECU_CODE', 'DAILY_WEIGHTED']],
                                      on=['Date', 'SECU_CODE'], how='left')
        self.pv_df['VWAP'] = self.pv_df['DAILY_WEIGHTED']

    def _update_daily_data(self):
        self.daily_data = self.pv_df.groupby(['Date', 'SECU_CODE']).last().reset_index()

    def _calculate_normalized_alpha(self, alpha_col, rank_col='rank', normalized_col='normalized_alpha'):
        self.daily_data[rank_col] = self.daily_data.groupby('Date')[alpha_col].rank()
        self.daily_data[normalized_col] = self.daily_data.groupby('Date')[rank_col].transform(
            lambda x: 2 * (x - 1) / (len(x) - 1) - 1 if len(x) > 1 else 0)

    def _calculate_industry_neutral_alpha(self, normalized_col, industry_neutral_col):
        self.daily_data[industry_neutral_col] = self.daily_data[normalized_col] - \
                                                self.daily_data.groupby('industry')[normalized_col].transform('mean')

    def _calculate_industry_size_neutral_alpha(self, size_col, industry_neutral_col, size_neutral_col):
        self.daily_data['log_size'] = np.log(self.daily_data['TRADABLE_SHARES'] * self.daily_data['cp'])
        latest_date = self.daily_data['Date'].max()
        latest_data = self.daily_data[self.daily_data['Date'] == latest_date].copy()
        print(latest_data)

        if not latest_data.empty:
            residuals = self._calculate_residuals(latest_data, size_col, industry_neutral_col)
            latest_data[size_neutral_col] = residuals

            # Ensure self.latest_data has the correct indices and initialize columns if they don't exist
            if self.latest_data.empty:
                self.latest_data = pd.DataFrame(index=self.daily_data.index)

            if size_neutral_col not in self.latest_data.columns:
                self.latest_data[size_neutral_col] = np.nan

            # Reindex self.latest_data to ensure it matches latest_data's indices
            self.latest_data = self.latest_data.reindex(index=latest_data.index, fill_value=np.nan)

            # Assign residuals to self.latest_data
            self.latest_data[size_neutral_col] = latest_data[size_neutral_col]

            # Update daily_data with the calculated values
            self.daily_data.loc[self.daily_data['Date'] == latest_date, size_neutral_col] = latest_data[
                size_neutral_col]
        else:
            print("Latest data is empty. Skipping residuals calculation.")

    def _calculate_combined_alpha(self):
        # Ensure the necessary columns are calculated before this function
        self.daily_data['combined_alpha'] = -1 * zscore(
            self.daily_data['industry_size_neutral_main_fund_alpha'].fillna(0)) + zscore(
            self.daily_data['industry_size_neutral_volume_alpha'].fillna(0))
        latest_date = self.daily_data['Date'].max()

        # Ensure self.latest_data has the correct indices and initialize columns if they don't exist
        if self.latest_data.empty:
            self.latest_data = pd.DataFrame(index=self.daily_data.index)

        if 'combined_alpha' not in self.latest_data.columns:
            self.latest_data['combined_alpha'] = np.nan

        # Reindex self.latest_data to ensure it matches the latest data's indices
        self.latest_data = self.latest_data.reindex(index=self.daily_data.index, fill_value=np.nan)

        # Assign combined_alpha to self.latest_data
        self.latest_data['combined_alpha'] = self.daily_data.loc[
            self.daily_data['Date'] == latest_date, 'combined_alpha']

    def _calculate_residuals(self, group, vector_column, response_column):
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

    def _ensure_column_exists(self, column_name):
        if column_name not in self.latest_data.columns:
            self.latest_data[column_name] = np.nan



    def simple_backtest(self, vector, initial_capital=1e8):
            df = self.daily_data.copy()
            df = df.sort_values(by='Date')
            if 'ADJ_CLOSE_PRICE' not in df.columns:
                print("ADJ_CLOSE_PRICE column is missing.")
                return None, None

            # Ensure there are no zero prices to avoid division by zero
            df['ADJ_CLOSE_PRICE'] = df['ADJ_CLOSE_PRICE'].replace(0, np.nan)
            df['ADJ_CLOSE_PRICE'] = df['ADJ_CLOSE_PRICE'].ffill()

            # Append the vector to the DataFrame to align by date and stock code, assuming vector is correctly indexed
            df['vector'] = df[vector]
            df['vector'] = df['vector'].fillna(0)

            # Ensure the vector column is numeric
            df['vector'] = df['vector'].astype(float)

            def weight_assignment(df):
                df = df.sort_values(by='Date')

                # Define masks for long and short investments based on the normalized factor
                df['long_weight'] = 0.0
                df['short_weight'] = 0.0

                df.loc[df['vector'] < 0, 'long_weight'] = -1* abs(df['vector']) / \
                                                           df[df['vector'] < 0].groupby('Date')[
                                                               'vector'].transform('sum')
                df.loc[df['vector'] >= 0, 'short_weight'] = -1*abs(df['vector']) / \
                                                           df[df['vector'] >= 0].groupby('Date')['vector'].transform('sum')

                df.loc[df['vector'] < 0, 'weight'] = df['long_weight']
                df.loc[df['vector'] >=0, 'weight'] = df['short_weight']
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

            # Calculate the next-day price change
            df['next_day_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].diff()
            df['next_day_return'] = df['next_day_return'].fillna(0)  # Fill NaNs that result from diff and shift

            # Shift investments to get the previous day's investments
            df['previous_investment'] = df.groupby('SECU_CODE')['investment'].shift(1)

            # Calculate investment changes
            df['investment_change'] = (df['investment'] - df['previous_investment']).fillna(0)
            df['abs_investment_change']= abs(df['investment_change'])

            # Calculate hold_pnl based on the condition
            condition = df['previous_investment'] * df['investment'] > 0
            df['hold_pnl'] = np.where(condition, df['previous_investment'] * df['next_day_return'], 0)
            df['adj_factor']= df['ADJ_CLOSE_PRICE']/df['cp']
            df['trade_pnl'] = df['investment_change'] * (df['ADJ_CLOSE_PRICE'] - df['adj_factor']*df['VWAP'])
            df['pnl'] = df['hold_pnl'].fillna(0) + df['trade_pnl'].fillna(0)
            # Calculate TVR Shares and TVR Values
            df['tvr_shares'] = df['abs_investment_change']
            df['tvr_values'] = df['abs_investment_change'] * df['ADJ_CLOSE_PRICE']
            df['tvr_shares'].fillna(0, inplace=True)
            df['tvr_values'].fillna(0, inplace=True)


            aggregated = df.groupby('Date').agg(
                pnl=('pnl', 'sum'),
                long_size=(
                    'investment',
                    lambda x: (x[df[vector].loc[x.index] >= 0] * df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum()),
                short_size=(
                    'investment',
                    lambda x: (-x[df[vector].loc[x.index] < 0] * df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum()),
                total_size=(
                    'investment',
                    lambda x: (x[df[vector].loc[x.index] >= 0] * df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum() +
                              (-x[df[vector].loc[x.index] < 0] * df.loc[x.index, 'ADJ_CLOSE_PRICE']).sum()),
                tvrshares=('tvr_shares', 'sum'),
                tvrvalues=('tvr_values', 'sum'),
                long_count=('vector', lambda x: (
                    (x >= x.shift(1))).sum()),
                short_count=('vector', lambda x: (
                    (x < x.shift(1))).sum()),
            ).reset_index()


            df['stocks_return'] = np.log(df['ADJ_CLOSE_PRICE'] / df['ADJ_VWAP'])

            # Calculate Information Coefficient (IC)
            aggregated['IC'] = aggregated['TRADINGDAY_x'].apply(
                lambda day: vector.corr(
                    df[df['TRADINGDAY_x'] == day]['stocks_return'])
            )
            daily_pnl = df['pnl'].sum()

            print(df['pnl'])
            print(daily_pnl)

            self.daily_data= df
            latest_date = df['Date'].max()
            latest_data = df[df['Date'] == latest_date].copy()
            self.latest_data= latest_data

            return df

if __name__ == "__main__":
    base_paths = [
        "/Users/zw/Desktop/TDBBase2021",
        "/Users/zw/Desktop/TDBBase2022",
        "/Users/zw/Desktop/TDBBase2023"
    ]
    sk_base_path = "/Users/zw/Desktop/DataBase/Sk"
    output_dir = "/Users/zw/Desktop"
    mv_liq_free_path = "/Users/zw/Desktop/DataBase-1/MvLiqFree"  # Path to the MvLiqFree files
    cp_base_path = '/Users/zw/Desktop/DataBase/CP'
    from_scratch = True  # Set to False if you want to start from the saved long format data
    industry_file_path = '/Users/zw/Desktop/IndustryCitic_with_industry.xlsx'

    combiner = DataCombiner(base_paths, sk_base_path,
                            mv_liq_free_path=mv_liq_free_path,cp_base_path=cp_base_path,industry_file_path=industry_file_path)

    # Combine files for L1MinuteCP and L1MinuteV data
    combiner.read_and_concatenate('L1MinuteCP', 'L1MinuteCP_', 'L1MinuteV', 'L1MinuteV_', max_files=10)
    # Concatenate the list of DataFrames into a single DataFrame
    results = pd.concat(combiner.all_results, ignore_index=True)

    print(results)
    results.to_csv('/Users/zw/Desktop/combined_alpha.csv', index=False)
    print('Results saved to file')

    # Release memory
    gc.collect()
