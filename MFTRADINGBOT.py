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
        self.aggregated= pd.DataFrame


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

                        if files_processed >= self.max_length-1:
                            # Process the queue
                            self.pv_df = self.queue_to_dataframe()
                            self.calculate_volume_critical()
                            self.merge_total_share()

                            # Calculate combined alphas
                            self.calculate_combined_alpha()
                            # self.simple_backtest('normalized_combined_alpha')

                            # Create directory if it doesn't exist
                            results_dir = "/Users/zw/Desktop/combined_alpha_results"

                            if not os.path.exists(results_dir):
                                os.makedirs(results_dir)

                            # Concatenate latest_data to DataFrame and save as CSV

                            date_str = self.pv_df['Date'].max().strftime('%Y%m%d')
                            output_file = os.path.join(results_dir, f"combined_alpha_{date_str}.csv")
                            # aggregate_file= os.path.join(results_dir, f"aggregated_combined_alpha_{date_str}.csv")
                            # Ensure self.latest_data is a DataFrame and save to CSV

                            self.latest_data.to_csv(output_file, index=False)
                            # self.aggregated.to_csv(aggregate_file, index=False)

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
        # Initialize a list to store merged data for all dates
        merged_data_list = []

        # Ensure the 'Date' column is in datetime format
        self.pv_df['Date'] = pd.to_datetime(self.pv_df['Date'])

        # Get unique dates
        unique_dates = self.pv_df['Date'].unique()

        # Iterate over each unique date
        for date in unique_dates:
            date_str = date.strftime('%Y%m%d')

            # Define file paths for MvLiqFree and CP files
            mv_file = os.path.join(self.mv_liq_free_path, f'MvLiqFree_{date_str}.txt')
            cp_file = os.path.join(self.cp_base_path, f'CP_{date_str}.txt')

            if os.path.exists(mv_file) and os.path.exists(cp_file):
                # Read MvLiqFree data
                liq_free_data = pd.read_csv(mv_file, delimiter='\t', header=None, encoding='utf-8')

                # Read CP data
                cp_data = pd.read_csv(cp_file, delimiter='\t', header=None, encoding='utf-8')

                # Get the unique SECU_CODEs for the given date
                secu_codes = self.pv_df.loc[self.pv_df['Date'] == date, 'SECU_CODE'].unique()

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
        pv_df = self.pv_df.set_index(['Date', 'SECU_CODE'])
        merged_data_combined = merged_data_combined.set_index(['Date', 'SECU_CODE'])

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
        self._calculate_main_fund_volume_alpha()


        self._update_daily_data()
        latest_date = self.daily_data['Date'].max()
        latest_data = self.daily_data[self.daily_data['Date'] == latest_date].copy()
        self.latest_data=latest_data

        print(self.latest_data)

    def _prepare_data(self):
        self.pv_df.sort_values(by=['Date', 'SECU_CODE'], inplace=True)
        self.pv_df['Date'] = pd.to_datetime(self.pv_df['Date'], errors='coerce')

    def _calculate_main_fund_weighted(self):
        df_filtered = self._filter_data_by_volume_and_time(self.pv_df)
        df_grouped = self._calculate_weighted_price(df_filtered)
        self.pv_df = self._merge_with_main_fund_weighted(self.pv_df, df_grouped)
        self._calculate_vwap()
        # Calculate main_fund_alpha
        self.pv_df['not_ma_main_fund_alpha'] = np.where(self.pv_df['VWAP'] != 0,
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
        df['not_ma_raw_volume_alpha'] = np.where(df['TRADABLE_SHARES'] != 0,
                                                 (df['support_volume'] - df['res_volume']) / df['TRADABLE_SHARES'], 0)

        self.pv_df = df

    def _calculate_main_fund_volume_alpha(self):
        df_filtered = self._filter_data_by_volume_and_time(self.pv_df)

        # Filter out minutes of main fund that is smaller than the weighted price
        df_filtered_sup = df_filtered[df_filtered['cp'] <= df_filtered['MAIN_FUND_WEIGHTED']].copy()
        df_sup = df_filtered_sup.groupby(['SECU_CODE', 'Date'])['v'].sum().reset_index().rename(
            columns={'v': 'main_fund_sup'})

        # Filter out minutes of main fund that is greater than the weighted price
        df_filtered_res = df_filtered[df_filtered['cp'] > df_filtered['MAIN_FUND_WEIGHTED']].copy()
        df_res = df_filtered_res.groupby(['SECU_CODE', 'Date'])['v'].sum().reset_index().rename(
            columns={'v': 'main_fund_res'})

        # Merge the support and resistance volumes back into the original dataframe
        df_filtered = df_filtered.merge(df_sup, on=['SECU_CODE', 'Date'], how='left')
        df_filtered = df_filtered.merge(df_res, on=['SECU_CODE', 'Date'], how='left')

        # Calculate main_fund_volume_alpha
        df_filtered['main_fund_volume_alpha'] = np.where(df_filtered['TRADABLE_SHARES'] != 0,
                                                         (df_filtered['main_fund_sup'] - df_filtered['main_fund_res']) /
                                                         df_filtered['TRADABLE_SHARES'], 0)

        print(df_filtered['main_fund_res'])
        print(df_filtered['main_fund_sup'])
        print(df_filtered['main_fund_volume_alpha'])

        self.pv_df = self.pv_df.merge(df_filtered[['SECU_CODE', 'Date', 'main_fund_volume_alpha']],
                                      on=['SECU_CODE', 'Date'], how='left')
        print(self.pv_df.columns)

    def _filter_data_by_volume_and_time(self, df):
        time_condition = (df['minute'] > 1) & (df['minute'] < 237)
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
        print("Updating daily data...")
        self.daily_data = self.pv_df.groupby(['Date', 'SECU_CODE']).last().reset_index()
        print("Daily data updated.")


    def _ensure_column_exists(self, column_name):
        if column_name not in self.latest_data.columns:
            self.latest_data[column_name] = np.nan


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
    combiner.read_and_concatenate('L1MinuteCP', 'L1MinuteCP_', 'L1MinuteV', 'L1MinuteV_', max_files=1e8)
    # Concatenate the list of DataFrames into a single DataFrame


    # Release memory
    gc.collect()
