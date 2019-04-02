import pandas as pd
import numpy as np
import scipy as sp
import bottleneck as bn
import itertools as it


###############################################################################
###############################################################################
class IndustryClassification:

    #--------------------------------------------------------------------------
    def __init__(self, industry_class_indicator, if_financial_futures,
                 if_get_inverse_classes):
        self.industry_class_indicator = industry_class_indicator
        self.if_financial_futures = if_financial_futures
        self.if_get_inverse_classes = if_get_inverse_classes

        self.industry_class_1 = {
            'PreciousMetal': ['au', 'ag'],
            'IndustrialMetal': ['cu', 'zn', 'al', 'ni', 'pb', 'sn'],
            'BlackChain': ['rb', 'j', 'i', 'jm', 'hc', 'SM', 'SF', 'FG', 'ZC'],
            'EnergyChem': ['l', 'MA', 'pp', 'TA', 'ru', 'bu', 'v', 'sc', 'eg'],
            'Agriculture': ['cs', 'c', 'a', 'm', 'RM', 'y', 'p', 'OI', 'b'],
            'SoftComm': ['CF', 'SR', 'jd', 'AP', 'sp']}

        self.industry_class_2 = {
            'Metal': ['cu', 'zn', 'al', 'ni', 'pb', 'sn', 'au', 'ag'],
            'BlackChain': ['rb', 'j', 'i', 'jm', 'hc', 'SM', 'SF'],
            'EnergyChem': ['l', 'MA', 'pp', 'TA', 'ru', 'bu', 'v',
                           'sc', 'ZC', 'FG', 'eg'],
            'Agri': ['CF', 'SR', 'a', 'm', 'RM', 'y', 'p',
                     'OI', 'cs', 'c', 'b'],
            'SoftComm': ['jd', 'AP', 'sp']}

        self.industry_class_3 = {
            'Indust': ['cu', 'zn', 'al', 'ni', 'pb', 'sn', 'au', 'ag', 'rb',
                       'j', 'i', 'jm', 'hc', 'SM', 'SF', 'FG', 'l', 'MA',
                       'pp', 'TA', 'ru', 'bu', 'v', 'sc', 'ZC', 'eg', 'sp'],
            'Agri': ['a', 'm', 'RM', 'y', 'p', 'OI', 'cs', 'c', 'CF',
                     'SR', 'jd', 'AP', 'b']}

        self.financial_futures = {
            'Index': ['IF', 'IH', 'IC'],
            'Rate': ['TF', 'T', 'TS']}

        self.reverse_industry_class = {}

    #--------------------------------------------------------------------------
    def get_industry_classes(self):
        if self.industry_class_indicator == 'industry_class_1':
            final_industry_class = self.industry_class_1
        elif self.industry_class_indicator == 'industry_class_2':
            final_industry_class = self.industry_class_2
        elif self.industry_class_indicator == 'industry_class_3':
            final_industry_class = self.industry_class_3
        else:
            final_industry_class = {}

        if self.if_financial_futures == True:
            final_industry_class.update(self.financial_futures)

        self.final_industry_class = final_industry_class

        return final_industry_class

    #--------------------------------------------------------------------------
    def get_inverse_industry_classes(self):
        if self.if_get_inverse_classes:
            if len(self.final_industry_class) > 0:
                list0 = sum(
                    [list(it.product([x], self.final_industry_class[x]))
                     for x in self.final_industry_class.keys()], [])
                self.reverse_industry_class = dict(
                    [[a[1], a[0]] for a in list0])

        return self.reverse_industry_class


###############################################################################
###############################################################################
class Factors:

    #--------------------------------------------------------------------------
    def __init__(self, data, daily_data = True):
        self.open = data['open']
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.pre_close = data['close'].shift()
        self.volume = data['volume']

        # If the data is daily-based, then the DataFrame contains the
        # columns of vwap and amount. But if the data is intraday,
        # the DataFrame may not contain the two columns. Therefore we
        # need to give the indicator whether the data is daily or not,
        # in case of class initiation error.
        if daily_data:
            self.vwap = data['vwap']
            self.amount = data['amount']

        self.index = data.index
        self.columns = self.close.columns

        self.industry_class_1 = {
            'PreciousMetal': ['au', 'ag'],
            'IndustrialMetal': ['cu', 'zn', 'al', 'ni', 'pb', 'sn'],
            'BlackChain': ['rb', 'j', 'i', 'jm', 'hc', 'SM', 'SF', 'FG', 'ZC'],
            'EnergyChem': ['l', 'MA', 'pp', 'TA', 'ru', 'bu', 'v', 'sc', 'eg'],
            'Agriculture': ['cs', 'c', 'a', 'm', 'RM', 'y', 'p', 'OI', 'b'],
            'SoftComm': ['CF', 'SR', 'jd', 'AP', 'sp']}

        self.industry_class_2 = {
            'Metal': ['cu', 'zn', 'al', 'ni', 'pb', 'sn', 'au', 'ag'],
            'BlackChain': ['rb', 'j', 'i', 'jm', 'hc', 'SM', 'SF'],
            'EnergyChem': ['l', 'MA', 'pp', 'TA', 'ru', 'bu', 'v',
                           'sc', 'ZC', 'FG', 'eg'],
            'Agri': ['CF', 'SR', 'a', 'm', 'RM', 'y', 'p',
                     'OI', 'cs', 'c', 'b'],
            'SoftComm': ['jd', 'AP', 'sp']}

        self.industry_class_3 = {
            'Indust': ['cu', 'zn', 'al', 'ni', 'pb', 'sn', 'au', 'ag', 'rb',
                       'j', 'i', 'jm', 'hc', 'SM', 'SF', 'FG', 'l', 'MA',
                       'pp', 'TA', 'ru', 'bu', 'v', 'sc', 'ZC', 'eg', 'sp'],
            'Agri': ['a', 'm', 'RM', 'y', 'p', 'OI', 'cs', 'c', 'CF',
                     'SR', 'jd', 'AP', 'b']}

        self.financial_futures = {
            'Index': ['IF', 'IH', 'IC'],
            'Rate': ['TF', 'T', 'TS']}

    #--------------------------------------------------------------------------
    def time_series_rank(self, x):
        return bn.rankdata(x)[-1]

    #--------------------------------------------------------------------------
    def get_liquid_contract_data(self, data_df, liquid_contract_df):
        liquid_data_df = data_df.copy()

        for column in liquid_contract_df.columns:
            if column in liquid_data_df.columns:
                liquid_data_df.loc[:, column] = np.where(
                    liquid_contract_df.loc[liquid_data_df.index, column] == 0,
                    np.nan, liquid_data_df.loc[:, column])

        return liquid_data_df

    #--------------------------------------------------------------------------
    def industry_factors(self, industry_class_indicator,
                         if_financial_futures = False):
        if industry_class_indicator == 'industry_class_1':
            final_industry_class = self.industry_class_1
        elif industry_class_indicator == 'industry_class_2':
            final_industry_class = self.industry_class_2
        elif industry_class_indicator == 'industry_class_3':
            final_industry_class = self.industry_class_3
        else:
            final_industry_class = {}

        if if_financial_futures == True:
            final_industry_class.update(self.financial_futures)

        industry_factors_dict = {}
        for industry, commodities in final_industry_class.items():
            use_commodities = [ele for ele in commodities
                               if ele in self.columns]
            factor_df = self.close * 0
            factor_df[use_commodities] = 1
            industry_factors_dict[industry] = factor_df

        return industry_factors_dict

    #--------------------------------------------------------------------------
    def rsi(self, n = 14):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_price = self.close[column].dropna()

            delta = column_price.diff()
            dUp, dDown = delta.copy(), delta.copy()
            dUp[dUp < 0] = 0
            dDown[dDown > 0] = 0

            RolUp = dUp.rolling(window = n).mean()
            RolDown = dDown.rolling(window = n).mean().abs()
            rs = RolUp / RolDown
            rsi = 100.0 - (100.0 / (1.0 + rs))

            alpha_df[column] = rsi.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def ma_close_ratio(self, window = 20):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            ma_close = column_close.rolling(window).mean()
            ma_close_ratio = ma_close / column_close

            alpha_df[column] = ma_close_ratio.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_001(self, rolling_window, liquid_contract_df):
        tmp_1_df = pd.DataFrame(index = self.index)
        tmp_2_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_volume = self.volume[column].dropna()
            column_close = self.close[column].dropna()
            column_open = self.open[column].dropna()

            tmp_1 = column_volume.diff()
            tmp_2 = (column_close - column_open) / column_open

            tmp_1_df[column] = tmp_1.reindex(tmp_1_df.index)
            tmp_2_df[column] = tmp_2.reindex(tmp_2_df.index)

        tmp_1_liquid = self.get_liquid_contract_data(
            tmp_1_df, liquid_contract_df)
        tmp_2_liquid = self.get_liquid_contract_data(
            tmp_2_df, liquid_contract_df)

        data1 = tmp_1_liquid.rank(axis = 1, pct = True)
        data2 = tmp_2_liquid.rank(axis = 1, pct = True)

        alpha_df = -data1.rolling(window = rolling_window).corr(
            data2, pairwise = False)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_002(self, delay_window = 1):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_low = self.low[column].dropna()
            column_high = self.high[column].dropna()

            column_result = (
                ((column_close - column_low) - (column_high - column_close))
                / (column_high - column_low)).diff(delay_window)

            alpha_df[column] = column_result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_003(self, delay_window = 1, rolling_window = 6):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            delay1 = column_close.shift(delay_window)
            column_low = self.low[column].dropna()

            condition1 = (column_close == delay1)
            condition2 = (column_close > delay1)
            condition3 = (column_close < delay1)

            part2 = (np.log(column_close)
                     - np.log(np.minimum(delay1[condition2],
                                         column_low[condition2])))
            part3 = (np.log(column_close)
                     - np.log(np.maximum(delay1[condition3],
                                         column_low[condition3])))

            result = part2.fillna(0) + part3.fillna(0)
            alpha = result.rolling(window = rolling_window).sum()

            alpha_df[column] = alpha.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_004(self, short_window = 2, long_window = 8, volume_window = 20):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            condition1 = (
                (column_close.rolling(window = long_window).mean()
                 + column_close.rolling(window = long_window).std())
                < (column_close.rolling(window = short_window).mean()))
            condition2 = (
                (column_close.rolling(window = short_window).mean())
                < (column_close.rolling(window = long_window).mean()
                   - column_close.rolling(window = long_window).std()))
            condition3 = (
                1 <= (column_volume
                      / column_volume.rolling(window = volume_window).mean()))

            indicator1 = pd.Series(np.ones(column_close.shape),
                                   index = column_close.index)
            indicator2 = -pd.Series(np.ones(column_close.shape),
                                    index = column_close.index)

            part1 = indicator2[condition1].reindex(column_close.index).fillna(0)
            part2 = (indicator1[~condition1][condition2]).reindex(
                column_close.index).fillna(0)
            part3 = (indicator1[~condition1][
                ~condition2][condition3]).reindex(column_close.index).fillna(0)
            part4 = (indicator2[~condition1][
                ~condition2][~condition3]).reindex(column_close.index).fillna(0)

            alpha = part1 + part2 + part3 + part4
            alpha_df[column] = alpha.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_004_alter(self, short_window = 2, long_window = 8,
                        rolling_sum_window = 20, volume_window = 20):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            condition1 = (
                (column_close.rolling(window = long_window).mean()
                 + column_close.rolling(window = long_window).std())
                < (column_close.rolling(window = short_window).mean()))
            condition2 = (
                (column_close.rolling(window = short_window).mean())
                < (column_close.rolling(window = long_window).mean()
                   - column_close.rolling(window = long_window).std()))
            condition3 = (
                1 <= (column_volume
                      / column_volume.rolling(window = volume_window).mean()))

            indicator1 = pd.Series(np.ones(column_close.shape),
                                   index = column_close.index)
            indicator2 = -pd.Series(np.ones(column_close.shape),
                                    index = column_close.index)

            part1 = indicator2[condition1].reindex(column_close.index).fillna(0)
            part2 = (indicator1[~condition1][condition2]).reindex(
                column_close.index).fillna(0)
            part3 = (indicator1[~condition1][
                ~condition2][condition3]).reindex(column_close.index).fillna(0)
            part4 = (indicator2[~condition1][
                ~condition2][~condition3]).reindex(column_close.index).fillna(0)

            alpha = part1 + part2 + part3 + part4
            alpha_df[column] = alpha.reindex(alpha_df.index)

        final_alpha_df = alpha_df.rolling(window = rolling_sum_window).sum()

        return final_alpha_df

    #--------------------------------------------------------------------------
    def alpha_005(self, rank_window = 5, corr_window = 3):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_volume = self.volume[column].dropna()
            column_high = self.high[column].dropna()

            ts_volume = column_volume.rolling(window = rank_window).apply(
                self.time_series_rank)
            ts_high = column_high.rolling(window = rank_window).apply(
                self.time_series_rank)

            corr_ts = ts_high.rolling(window = rank_window).corr(
                ts_volume, pairwise = False)

            alpha = corr_ts.rolling(window = corr_window).max()

            alpha_df[column] = alpha.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_006(self, liquid_contract_df, open_mult = 0.85, diff_window = 4):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_open = self.open[column].dropna()
            column_high = self.high[column].dropna()

            condition1 = ((column_open * open_mult
                           + column_high * (1 - open_mult)).diff(
                               diff_window) > 0)
            condition2 = ((column_open * open_mult
                           + column_high * (1 - open_mult)).diff(
                               diff_window) == 0)
            condition3 = ((column_open * open_mult
                           + column_high * (1 - open_mult)).diff(
                               diff_window) < 0)

            indicator1 = pd.Series(np.ones(column_open.shape),
                                   index = column_open.index)
            indicator2 = pd.Series(np.zeros(column_open.shape),
                                   index = column_open.index)
            indicator3 = -pd.Series(np.ones(column_open.shape),
                                    index = column_open.index)

            part1 = indicator1[condition1].reindex(column_open.index).fillna(0)
            part2 = indicator2[condition2].reindex(column_open.index).fillna(0)
            part3 = indicator3[condition3].reindex(column_open.index).fillna(0)

            result = part1 + part2 + part3
            alpha_df[column] = result.reindex(alpha_df.index)

        alpha_liquid = self.get_liquid_contract_data(
            alpha_df, liquid_contract_df)
        final_alpha_df = alpha_liquid.rank(axis = 1, pct = True)

        return final_alpha_df

    #--------------------------------------------------------------------------
    def alpha_006_alter(self, liquid_contract_df,
                        open_mult = 0.85, diff_window = 4):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_open = self.open[column].dropna()
            column_high = self.high[column].dropna()

            result = np.log(column_open* open_mult
                            + column_high * (1 - open_mult)).diff(diff_window)

            alpha_df[column] = result.reindex(alpha_df.index)

        alpha_liquid = self.get_liquid_contract_data(
            alpha_df, liquid_contract_df)
        final_alpha_df = alpha_liquid.rank(axis = 1, pct = True)

        return final_alpha_df

    #--------------------------------------------------------------------------
    def alpha_007(self, liquid_contract_df, com_num_1 = 3,
                  com_num_2 = 3, com_num_3 = 3):
        part_1_df = pd.DataFrame(index = self.index)
        part_2_df = pd.DataFrame(index = self.index)
        part_3_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_vwap = self.vwap[column].dropna()
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            part1 = np.maximum(column_vwap - column_close, com_num_1)
            part2 = np.minimum(column_vwap - column_close, com_num_2)
            part3 = column_volume.diff(com_num_3)

            part_1_df[column] = part1.reindex(part_1_df.index)
            part_2_df[column] = part2.reindex(part_2_df.index)
            part_3_df[column] = part3.reindex(part_3_df.index)

        part_1_liquid = self.get_liquid_contract_data(
            part_1_df, liquid_contract_df)
        part_2_liquid = self.get_liquid_contract_data(
            part_2_df, liquid_contract_df)
        part_3_liquid = self.get_liquid_contract_data(
            part_3_df, liquid_contract_df)

        part_1_rank = part_1_liquid.rank(axis = 1, pct = True)
        part_2_rank = part_2_liquid.rank(axis = 1, pct = True)
        part_3_rank = part_3_liquid.rank(axis = 1, pct = True)

        alpha_df = part_1_rank + part_2_rank * part_3_rank

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_007_alter(self, liquid_contract_df, diff_window = 3):
        part_1_df = pd.DataFrame(index = self.index)
        part_2_df = pd.DataFrame(index = self.index)
        part_3_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_vwap = self.vwap[column].dropna()
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            part1 = (np.log(column_vwap) - np.log(column_close)).rolling(
                window = diff_window).max()
            part2 = (np.log(column_vwap) - np.log(column_close)).rolling(
                window = diff_window).min()
            part3 = np.log(column_volume).diff(diff_window)

            part_1_df[column] = part1.reindex(part_1_df.index)
            part_2_df[column] = part2.reindex(part_2_df.index)
            part_3_df[column] = part3.reindex(part_3_df.index)

        part_1_liquid = self.get_liquid_contract_data(
            part_1_df, liquid_contract_df)
        part_2_liquid = self.get_liquid_contract_data(
            part_2_df, liquid_contract_df)
        part_3_liquid = self.get_liquid_contract_data(
            part_3_df, liquid_contract_df)

        part_1_rank = part_1_liquid.rank(axis = 1, pct = True)
        part_2_rank = part_2_liquid.rank(axis = 1, pct = True)
        part_3_rank = part_3_liquid.rank(axis = 1, pct = True)

        alpha_df = part_1_rank + part_2_rank * part_3_rank

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_008(self, liquid_contract_df,
                  high_low_mult = 0.2, diff_window = 4):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()
            column_vwap = self.vwap[column].dropna()

            temp = -np.log(
                (column_high + column_low) * 0.5 * high_low_mult
                + column_vwap * (1 - high_low_mult)).diff(diff_window)

            alpha_df[column] = temp.reindex(alpha_df.index)

        alpha_liquid = self.get_liquid_contract_data(
            alpha_df, liquid_contract_df)
        final_alpha_df = alpha_liquid.rank(axis = 1, pct = True)

        return final_alpha_df

    #--------------------------------------------------------------------------
    def alpha_009(self, alpha = 2.0 / 7.0):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()
            column_volume = self.volume[column].dropna()

            temp = (
                ((column_high + column_low) * 0.5
                 - (column_high.shift() + column_low.shift()) * 0.5)
                * (column_high - column_low) / column_volume)
            alpha_df[column] = temp.ewm(alpha = alpha).mean().reindex(
                alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_010(self, liquid_contract_df, std_window = 20, com_num = 5):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_ret = np.log(column_close).diff()
            condition = (column_ret < 0)

            part1 = (column_ret.rolling(
                window = std_window).std()[condition]).reindex(
                    column_ret.index).fillna(0)
            part2 = (column_close[~condition]).reindex(
                column_ret.index).fillna(0)

            result = np.maximum((part1 + part2) ** 2, com_num)
            alpha_df[column] = result.reindex(alpha_df.index)

        alpha_liquid = self.get_liquid_contract_data(
            alpha_df, liquid_contract_df)
        final_alpha_df = alpha_liquid.rank(axis = 1, pct = True)

        return final_alpha_df

    #--------------------------------------------------------------------------
    def alpha_010_alter(self, liquid_contract_df,
                        std_window = 20, com_num = 5):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_ret = np.log(column_close).diff()
            condition = (column_ret < 0)

            part1 = (column_ret.rolling(
                window = std_window).std()[condition]).reindex(
                    column_ret.index).fillna(0)
            part2 = (column_close[~condition]).reindex(
                column_close.index).fillna(0)

            result = ((part1 + part2) ** 2).rolling(window = com_num).max()
            alpha_df[column] = result.reindex(alpha_df.index)

        alpha_liquid = self.get_liquid_contract_data(
            alpha_df, liquid_contract_df)
        final_alpha_df = alpha_liquid.rank(axis = 1, pct = True)

        return final_alpha_df

    #--------------------------------------------------------------------------
    def alpha_011(self, rolling_window = 6):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_low = self.low[column].dropna()
            column_high = self.high[column].dropna()
            column_volume = self.volume[column].dropna()

            temp = (((column_close - column_low)
                     - (column_high - column_close))
                    / (column_high - column_low))
            result = temp * column_volume

            alpha_df[column] = result.rolling(
                window = rolling_window).sum().reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_011_alter(self, rolling_window = 6):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_low = self.low[column].dropna()
            column_high = self.high[column].dropna()

            temp = (((column_close - column_low)
                     - (column_high - column_close))
                    / (column_high - column_low))

            alpha_df[column] = temp.rolling(
                window = rolling_window).sum().reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_012(self, liquid_contract_df, vwap_window = 10):
        temp_1_df = pd.DataFrame(index = self.index)
        temp_2_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_vwap = self.vwap[column].dropna()
            column_open = self.open[column].dropna()
            column_close = self.close[column].dropna()

            column_vwap_ma = column_vwap.rolling(window = vwap_window).mean()
            temp_1 = column_open - column_vwap_ma
            temp_1_df[column] = temp_1.reindex(temp_1_df.index)

            temp_2 = (column_close - column_vwap).abs()
            temp_2_df[column] = temp_2.reindex(temp_2_df.index)

        temp_1_liquid = self.get_liquid_contract_data(
            temp_1_df, liquid_contract_df)
        temp_2_liquid = self.get_liquid_contract_data(
            temp_2_df, liquid_contract_df)

        part1 = temp_1_liquid.rank(axis = 1, pct = True)
        part2 = -temp_2_liquid.rank(axis = 1, pct = True)

        alpha_df = (part1 * part2)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_013(self):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()
            column_vwap = self.vwap[column].dropna()

            result = ((column_high * column_low) ** 0.5) - column_vwap
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_013_alter(self):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()
            column_vwap = self.vwap[column].dropna()

            result = (np.log((column_high * column_low) ** 0.5)
                      - np.log(column_vwap))
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_014(self, shift_window = 5):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            result = column_close - column_close.shift(shift_window)
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_014_alter(self, shift_window = 5):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            result = (np.log(column_close)
                      - np.log(column_close.shift(shift_window)))
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_015(self, close_shift = 1):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_open = self.open[column].dropna()
            column_close = self.close[column].dropna()

            result = column_open / column_close.shift(close_shift) - 1
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_016(self, liquid_contract_df, corr_window = 5, max_window = 5):
        '''
        This feature might not be suitable for intraday data.
        '''
        volume_liquid = self.get_liquid_contract_data(
            self.volume, liquid_contract_df)
        vwap_liquid = self.get_liquid_contract_data(
            np.log(self.vwap).diff(), liquid_contract_df)

        temp1 = volume_liquid.rank(axis = 1, pct = True)
        temp2 = vwap_liquid.rank(axis = 1, pct = True)

        part = temp1.rolling(window = corr_window).corr(
            temp2, pairwise = False)
        part = part[(part < np.inf) & (part > -np.inf)]

        result = part.rank(axis = 1, pct = True)
        alpha = result.rolling(window = max_window).max().dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_017(self, liquid_contract_df,
                  ts_max_window = 15, close_diff_window = 5):
        part_1_df = pd.DataFrame(index = self.index)
        part_2_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_vwap = self.vwap[column].dropna()
            column_close = self.close[column].dropna()

            column_vwap_rolling = column_vwap.rolling(
                window = ts_max_window).max()
            temp_1 = (column_vwap - column_vwap_rolling)
            part_1_df[column] = temp_1.reindex(part_1_df.index)

            temp_2 = np.log(column_close).diff(close_diff_window)
            part_2_df[column] = temp_2.reindex(part_2_df.index)

        part_1_liquid = self.get_liquid_contract_data(
            part_1_df, liquid_contract_df)
        part_1_rank = part_1_liquid.rank(axis = 1, pct = True)

        alpha_df = (part_1_rank ** part_2_df)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_018(self, delay_window = 5):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            delay_close = column_close.shift(delay_window)

            alpha = column_close / delay_close
            alpha_df[column] = alpha.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_019(self, delay_window = 5):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_delay = column_close.shift(delay_window)

            condition_1 = column_close < column_delay
            condition_3 = column_close > column_delay

            part_1 = ((column_close[condition_1] - column_delay[condition_1])
                      / column_delay[condition_1])
            part_1 = part_1.reindex(column_close.index).fillna(0)

            part_2 = ((column_close[condition_3] - column_delay[condition_3])
                      / column_close[condition_3])
            part_2 = part_2.reindex(column_close.index).fillna(0)

            result = part_1 + part_2
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_020(self, delay_window = 6):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_delay = column_close.shift(delay_window)

            result = (column_close - column_delay) * 100 / column_delay
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_021(self, close_rolling_window = 6, minimum_estimate_size = 6):
        '''
        The calculation process of this factor may need careful check.
        '''
        alpha_df = pd.DataFrame(index = self.index)

        part_2 = np.arange(1, close_rolling_window + 1)
        for column in self.columns:
            column_close = self.close[column].dropna()
            part_1 = column_close.rolling(
                window = close_rolling_window).mean()

            N = part_1.shape[0]
            date_list = [
                [part_1.index[i-close_rolling_window+1], part_1.index[i]]
                for i in range(close_rolling_window-1, N)]

            beta_df = pd.DataFrame(
                index = part_1.index[close_rolling_window:])

            estimate_df = pd.DataFrame(index = part_1.index)
            estimate_df[column] = part_1
            estimate_df['const'] = 1

            for date_pair in date_list:
                start_date = date_pair[0]
                end_date = date_pair[1]

                data_df = estimate_df.loc[start_date:end_date]
                data_df['x'] = part_2.copy()
                data_df = data_df.dropna()

                if data_df.shape[0] >= minimum_estimate_size:
                    x = np.array(data_df.loc[:, ['x', 'const']])
                    x = x.reshape((len(x), 2))
                    y = np.array(data_df.loc[:, column])

                    beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
                    beta_df.loc[end_date, column] = beta[0]
                else:
                    beta_df.loc[end_date, column] = np.nan

            alpha_df[column] = beta_df.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_021_alter(self, close_rolling_window = 6):
        alpha_df = pd.DataFrame(index = self.index)

        part_2 = np.arange(1, close_rolling_window + 1)
        for column in self.columns:
            column_close = self.close[column].dropna()
            part_1 = column_close.rolling(
                window = close_rolling_window).mean()

            result = part_1.rolling(window = close_rolling_window).apply(
                lambda x: np.corrcoef(x, part_2)[0,1])
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_022(self, close_window = 6, shift_window = 3,
                  alpha = 1.0 / 12.0):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            part_1 = ((column_close
                       - column_close.rolling(window = close_window).mean())
                      / column_close.rolling(window = close_window).mean())
            temp = ((column_close
                     - column_close.rolling(window = close_window).mean())
                    / column_close.rolling(window = close_window).mean())
            part_2 = temp.shift(shift_window)

            result = (part_1 - part_2).ewm(alpha = alpha).mean()
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_023(self, shift_window = 1,
                  rolling_window = 20, alpha = 1.0/20.0):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            condition1 = (column_close > column_close.shift(shift_window))

            temp1 = column_close.rolling(
                window = rolling_window).std()[condition1]
            temp1 = temp1.fillna(0)

            temp2 = column_close.rolling(
                window = rolling_window).std()[~condition1]
            temp2 = temp2.fillna(0)

            part1 = temp1.ewm(alpha = alpha).mean()
            part2 = temp2.ewm(alpha = alpha).mean()

            result = part1 * 100 / (part1 + part2)
            alpha_df[column] = result.reindex(alpha_df.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_024(self, shift_window = 5, alpha = 1.0/5.0):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            delay = column_close.shift(shift_window)
            result = column_close - delay
            final_result = result.ewm(alpha = alpha).mean()

            alpha_df[column] = final_result

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_024_alter(self, shift_window = 5, alpha = 1.0/5.0):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            delay = self.close.shift(shift_window)
            result = np.log(column_close) - np.log(delay)
            final_result = result.ewm(alpha = alpha).mean()

            alpha_df[column] = final_result

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_025(self, liquid_contract_df, close_shift_window = 7,
                  linear_decay_window = 9, volume_window = 20,
                  mom_window = 250):
        part1 = pd.DataFrame(index = self.index)
        part2 = pd.DataFrame(index = self.index)
        part3 = pd.DataFrame(index = self.index)

        n = linear_decay_window
        linear_decay_seq = np.array([2*i/(n*(n+1)) for i in range(1, n+1)])

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            tmp_part1 = column_close - column_close.shift(close_shift_window)

            tmp_part2 = (column_volume
                         / column_volume.rolling(volume_window).mean())
            tmp_part2 = tmp_part2.rolling(linear_decay_window).apply(
                lambda x: np.sum(x * linear_decay_seq))

            tmp_part3 = np.log(column_close).diff().rolling(mom_window).sum()

            part1[column] = tmp_part1.reindex(self.index)
            part2[column] = tmp_part2.reindex(self.index)
            part3[column] = tmp_part3.reindex(self.index)

        part1_liquid = self.get_liquid_contract_data(part1, liquid_contract_df)
        part2_liquid = self.get_liquid_contract_data(part2, liquid_contract_df)
        part3_liquid = self.get_liquid_contract_data(part3, liquid_contract_df)

        part1_rank = part1_liquid.rank(axis = 1, pct = True)
        part2_rank = part2_liquid.rank(axis = 1, pct = True)
        part3_rank = part3_liquid.rank(axis = 1, pct = True)

        alpha_df = -part1_rank * (1 - part2_rank) * (1 + part3_rank)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_026(self, close_window = 7, shift_window = 5, vwap_window = 230):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_vwap = self.vwap[column].dropna()

            part1 = (column_close.rolling(window = close_window).mean()
                     - column_close)
            delay = column_close.shift(shift_window)
            part2 = column_vwap.rolling(window = vwap_window).corr(delay)

            alpha = part1 + part2
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_026_alter(self, close_window = 7,
                        shift_window = 5, vwap_window = 230):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_vwap = self.vwap[column].dropna()

            part1 = (
                np.log(column_close.rolling(window = close_window).mean())
                - np.log(column_close))
            delay = column_close.shift(shift_window)
            part2 = column_vwap.rolling(window = vwap_window).corr(delay)

            alpha = part1 + part2
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_028(self, rolling_window = 9, alpha = 1.0/3.0,
                  part1_multi = 3, part2_multi = 2):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()

            temp1 = (column_close
                     - column_low.rolling(window = rolling_window).min())
            temp2 = (column_high.rolling(window = rolling_window).max()
                     - column_low.rolling(window = rolling_window).min())
            part1 = (part1_multi
                     * (temp1 * 100 / temp2).ewm(alpha = alpha).mean())

            temp3 = (temp1 * 100 / temp2).ewm(alpha = alpha).mean()
            part2 = part2_multi * temp3.ewm(alpha = alpha).mean()

            result = part1 - part2
            alpha_df[column] = result.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_029(self, delay_window = 6):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            delay = column_close.shift(delay_window)
            result = (column_close - delay) * column_volume / delay

            alpha_df[column] = result.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_031(self, close_window = 12):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            result = (
                (column_close
                 - column_close.rolling(window = close_window).mean())
                / column_close.rolling(window = close_window).mean() * 100)

            alpha_df[column] = result.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_032(self, liquid_contract_df, corr_window = 3, sum_window = 3):
        alpha_df = pd.DataFrame(index = self.index)

        high_liquid = self.get_liquid_contract_data(
            self.high, liquid_contract_df)
        volume_liquid = self.get_liquid_contract_data(
            self.volume, liquid_contract_df)

        high_rank_df = high_liquid.rank(axis = 1, pct = True)
        volume_rank_df = volume_liquid.rank(axis = 1, pct = True)

        for column in self.columns:
            column_high_rank = high_rank_df[column].dropna()
            column_volume_rank = volume_rank_df[column].dropna()

            result = column_high_rank.rolling(corr_window).corr(
                column_volume_rank)
            result[result >= 1.0] = 1.0
            result[result <= -1.0] = -1.0
            result = result.fillna(0)

            alpha_df[column] = result.rolling(
                sum_window).sum().reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_033(self, liquid_contract_df, ts_min_window = 5,
                  ts_min_shift_window = 5, long_mom_window = 240,
                  short_mom_window = 20, ts_rank_window = 5):
        part_1_df = pd.DataFrame(index = self.index)
        part_2_df = pd.DataFrame(index = self.index)
        part_3_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_close_return = np.log(column_close).diff()
            column_low = self.low[column].dropna()
            column_volume = self.volume[column].dropna()

            column_ts_min_low = column_low.rolling(ts_min_window).min()
            part_1 = (np.log(column_ts_min_low.shift(ts_min_shift_window))
                      - np.log(column_ts_min_low))
            part_1_df[column] = part_1.reindex(self.index)

            column_mom = (
                (column_close_return.rolling(long_mom_window).sum()
                 - column_close_return.rolling(short_mom_window).sum())
                / float(long_mom_window - short_mom_window))
            part_2_df[column] = column_mom.reindex(self.index)

            part_3 = (
                column_volume.rolling(ts_rank_window).apply(
                    self.time_series_rank)
                / float(ts_rank_window))
            part_3_df[column] = part_3.reindex(self.index)

        part_1_liquid = self.get_liquid_contract_data(
            part_1_df, liquid_contract_df)
        part_2_liquid = self.get_liquid_contract_data(
            part_2_df, liquid_contract_df)
        part_3_liquid = self.get_liquid_contract_data(
            part_3_df, liquid_contract_df)

        alpha_df = (part_1_liquid
                    * part_2_liquid.rank(axis = 1, pct = True)
                    * part_3_liquid)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_034(self, close_window = 12):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            result = (column_close.rolling(window = close_window).mean()
                      / column_close)

            alpha_df[column] = result.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_035(self, liquid_contract_df, decay_window_1 = 15,
                  decay_window_2 = 7, corr_window = 17,
                  open_weight = 0.65, close_weight = 0.35):
        n = decay_window_1
        m = decay_window_1
        weight_1 = np.array([2*i/(n*(n+1)) for i in range(1, n+1)])
        weight_2 = np.array([2*i/(m*(m+1)) for i in range(1, m+1)])

        part_1_df = pd.DataFrame(index = self.index)
        part_2_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_open = self.open[column].dropna()
            column_open_return = np.log(column_open).diff()
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            part_1 = column_open_return.rolling(n).apply(lambda x: x*weight_1)
            part_1_df[column] = part_1.reindex(self.index)

            combine_price = (open_weight * column_open
                             + close_weight * column_close)
            corr_df = combine_price.rolling(corr_window).corr(column_volume)
            corr_df[corr_df >= 1.0] = 1.0
            corr_df[corr_df <= -1.0] = -1.0
            corr_df = corr_df.fillna(0)
            part_2 = corr_df.rolling(m).apply(lambda x: x*weight_2)
            part_2_df[column] = part_2.reindex(self.index)

        part_1_liquid = self.get_liquid_contract_data(
            part_1_df, liquid_contract_df)
        part_2_liquid = self.get_liquid_contract_data(
            part_2_df, liquid_contract_df)

        part_1_rank = part_1_liquid.rank(axis = 1, pct = True)
        part_2_rank = part_2_liquid.rank(axis = 1, pct = True)
        alpha_df = pd.concat(
            [part_1_rank, part_2_rank], axis = 0).min(level = 0)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_036(self, liquid_contract_df, corr_window = 6, sum_window = 2):
        volume_liquid = self.get_liquid_contract_data(
            self.volume, liquid_contract_df)
        vwap_liquid = self.get_liquid_contract_data(
            self.vwap, liquid_contract_df)

        volume_rank = volume_liquid.rank(axis = 1, pct = True)
        vwap_rank = vwap_liquid.rank(axis = 1, pct = True)

        total_corr_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_volume_rank = volume_rank[column].dropna()
            column_vwap_rank = vwap_rank[column].dropna()

            corr_df = column_volume_rank.rolling(
                corr_window).corr(column_vwap_rank)
            corr_df[corr_df >= 1.0] = 1.0
            corr_df[corr_df <= -1.0] = -1.0
            corr_df = corr_df.fillna(0)

            total_corr_df[column] = corr_df.rolling(
                sum_window).sum().reindex(self.index)

        alpha_df = total_corr_df.rank(axis = 1, pct = True)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_037(self, liquid_contract_df, mom_window = 5, shift_window = 10):
        total_mom_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_open = self.open[column].dropna()
            column_close = self.close[column].dropna()

            column_open_return = np.log(column_open).diff()
            column_close_return = np.log(column_close).diff()

            mom_df = (column_open_return.rolling(mom_window).sum()
                      * column_close_return.rolling(mom_window).sum())
            lag_mom_df = mom_df.shift(shift_window)

            total_mom_df[column] = (mom_df - lag_mom_df).reindex(self.index)

        total_mom_liquid = self.get_liquid_contract_data(
            total_mom_df, liquid_contract_df)
        alpha_df = total_mom_liquid.rank(axis = 1, pct = True)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_038(self, rolling_window = 20, diff_window = 2):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_high = self.high[column].dropna()

            sum_20 = (column_high.rolling(window = rolling_window).sum()
                      / float(rolling_window))
            delta2 = column_high.diff(diff_window)
            condition = (sum_20 < column_high)
            result = -delta2[condition].fillna(0)

            alpha_df[column] = result.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_039(self, liquid_contract_df,
                  weight_window_1 = 8, weight_window_2 = 12,
                  close_window = 2, vwap_weight = 0.3,
                  volume_mean_window = 180, volume_sum_window = 37,
                  corr_window = 14):
        n = weight_window_1
        m = weight_window_2
        weight_1 = np.array([2*i/(n*(n+1)) for i in range(1, n+1)])
        weight_2 = np.array([2*i/(m*(m+1)) for i in range(1, m+1)])

        part_1_df = pd.DataFrame(index = self.index)
        part_2_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_vwap = self.vwap[column].dropna()
            column_open = self.open[column].dropna()
            column_volume = self.volume[column].dropna()

            column_close_return = np.log(column_close).diff(close_window)

            part_1 = column_close_return.rolling(n).apply(
                lambda x: x*weight_1)
            part_1_df[column] = part_1.reindex(self.index)

            combined_price = (column_vwap * vwap_weight
                              + column_open * (1 - vwap_weight))
            volume_mean = column_volume.rolling(volume_mean_window).mean()
            volume_sum = volume_mean.rolling(volume_sum_window).sum()
            corr_df = combined_price.rolling(corr_window).corr(volume_sum)
            corr_df[corr_df >= 1.0] = 1.0
            corr_df[corr_df <= -1.0] = -1.0
            corr_df = corr_df.fillna(0)

            part_2 = corr_df.rolling(m).apply(lambda x: x*weight_2)
            part_2_df[column] = part_2.reindex(self.index)

        part_1_liquid = self.get_liquid_contract_data(
            part_1_df, liquid_contract_df)
        part_2_liquid = self.get_liquid_contract_data(
            part_2_df, liquid_contract_df)

        part_1_rank = part_1_liquid.rank(axis = 1, pct = True)
        part_2_rank = part_2_liquid.rank(axis = 1, pct = True)

        alpha_df = part_1_rank - part_2_rank

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_040(self, close_shift_window = 1, volume_window = 26):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            condition = (column_close > column_close.shift(close_shift_window))

            volume_up = column_volume[condition].fillna(0)
            volume_up_sum = volume_up.rolling(volume_window).sum()

            volume_down = column_volume[~condition].fillna(0)
            volume_down_sum = volume_down.rolling(volume_window).sum()

            alpha = 100 * volume_up_sum / float(volume_down_sum)
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_041(self, liquid_contract_df,
                  vwap_window = 3, price_threshold = 5):
        part_1_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_vwap = self.vwap[column].dropna()

            column_vwap_gap = column_vwap.diff(vwap_window)
            part_1 = np.maximum(column_vwap_gap, price_threshold)
            part_1_df[column] = part_1.reindex(self.index)

        part_1_liquid = self.get_liquid_contract_data(
            part_1_df, liquid_contract_df)
        alpha_df = part_1_liquid.rank(axis = 1, pct = True)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_042(self, liquid_contract_df, corr_window = 10, std_window = 10):
        part_1_df = pd.DataFrame(index = self.index)
        part_2_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_high = self.high[column].dropna()
            column_volume = self.volume[column].dropna()

            part_1 = column_high.rolling(corr_window).corr(column_volume)
            part_1[part_1 >= 1.0] = 1.0
            part_1[part_1 <= -1.0] = -1.0
            part_1 = part_1.fillna(0)
            part_1_df[column] = part_1.reindex(self.index)

            part_2 = column_high.rolling(std_window).std()
            part_2_df[column] = part_2.reindex(self.index)

        part_2_liquid = self.get_liquid_contract_data(
            part_2_df, liquid_contract_df)
        part_2_rank = part_2_liquid.rank(axis = 1, pct = True)

        alpha_df = -part_1_df * part_2_rank

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_043(self, close_window = 1, sum_window = 6):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            condition_1 = (column_close > column_close.shift(close_window))
            condition_2 = (column_close < column_close.shift(close_window))

            part_1 = column_volume[condition_1].fillna(0)
            part_2 = -column_volume[condition_2].fillna(0)

            result = part_1 + part_2
            alpha = result.rolling(sum_window).sum()
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_044(self, weight_window_1 = 6, weight_window_2 = 10,
                  volume_mean_window = 10, corr_window = 7,
                  ts_window_1 = 4, vwap_window = 3, ts_window_2 = 15):
        n = weight_window_1
        m = weight_window_2
        weight_1 = np.array([2*i/(n*(n+1)) for i in range(1, n+1)])
        weight_2 = np.array([2*i/(m*(m+1)) for i in range(1, m+1)])
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_low = self.low[column].dropna()
            column_volume = self.volume[column].dropna()
            column_vwap = self.vwap[column].dropna()

            volume_mean = column_volume.rolling(volume_mean_window).mean()
            tmp_1 = column_low.rolling(corr_window).corr(volume_mean)
            part_1 = tmp_1.rolling(n).apply(lambda x: x*weight_1)
            part_1_ts_rank = part_1.rolling(ts_window_1).apply(
                self.time_series_rank)

            vwap_diff = np.log(column_vwap).diff(vwap_window)
            part_2 = vwap_diff.rolling(m).apply(lambda x: x*weight_2)
            part_2_ts_rank = part_2.rolling(ts_window_2).apply(
                self.time_series_rank)

            alpha = part_1_ts_rank + part_2_ts_rank
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_045(self, liquid_contract_df,
                  close_weight = 0.6, shift_window = 1,
                  volume_mean_window = 150, corr_window = 15):
        part_1_df = pd.DataFrame(index = self.index)
        part_2_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_open = self.open[column].dropna()
            column_vwap = self.vwap[column].dropna()
            column_volume = self.volume[column].dropna()

            combined_price = (column_close * close_weight
                              + column_open * (1 - close_weight))
            part_1 = np.log(combined_price).diff(shift_window)
            part_1_df[column] = part_1.reindex(self.index)

            volume_mean = column_volume.rolling(volume_mean_window).mean()
            part_2 = column_vwap.rolling(corr_window).corr(volume_mean)
            part_2_df[column] = part_2.reindex(self.index)

        part_1_liquid = self.get_liquid_contract_data(
            part_1_df, liquid_contract_df)
        part_2_liquid = self.get_liquid_contract_data(
            part_2_df, liquid_contract_df)

        part_1_rank = part_1_liquid.rank(axis = 1, pct = True)
        part_2_rank = part_2_liquid.rank(axis = 1, pct = True)
        alpha_df = part_1_rank * part_2_rank

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_046(self, close_window_1 = 3, close_window_2 = 6,
                   close_window_3 = 12, close_window_4 = 24):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            close_mean_1 = column_close.rolling(close_window_1).mean()
            close_mean_2 = column_close.rolling(close_window_2).mean()
            close_mean_3 = column_close.rolling(close_window_3).mean()
            close_mean_4 = column_close.rolling(close_window_4).mean()

            alpha = (
                (close_mean_1 + close_mean_2 + close_mean_3 + close_mean_4)
                * 0.25 / column_close)
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_047(self, ts_rank_window = 6, alpha = 1.0/9.0):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()

            part_1 = column_high.rolling(ts_rank_window).max() - column_close
            part_2 = (column_high.rolling(ts_rank_window).max()
                      - column_low.rolling(ts_rank_window).min())
            alpha = (100 * part_1 / part_2).ewm(alpha = alpha).mean()

            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_048(self, liquid_contract_df,
                   volume_window_1 = 5, volume_window_2 = 20):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_volume = self.volume[column].dropna()

            part_1 = (np.sign(column_close - column_close.shift(1))
                      + np.sign(column_close.shift(1) - column_close.shift(2))
                      + np.sign(column_close.shift(2) - column_close.shift(3)))
            part_2 = (column_volume.rolling(volume_window_1).mean()
                      / column_volume.rolling(volume_window_2).mean())

            alpha = part_1 * part_2
            alpha_df[column] = alpha.reindex(self.index)

        alpha_liquid = self.get_liquid_contract_data(
            alpha_df, liquid_contract_df)
        alpha_rank = alpha_liquid.rank(axis = 1, pct = True)

        return alpha_rank

    #--------------------------------------------------------------------------
    def alpha_049(self, sum_window = 12):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()

            condition_1 = (
                (column_high + column_low)
                >= (column_high.shift() + column_low.shift()))
            condition_2 = (
                (column_high + column_low)
                <= (column_high.shift() + column_low.shift()))

            part_1 = np.maximum(np.abs(column_high - column_high.shift()),
                                np.abs(column_low - column_low.shift()))
            part_1[condition_1] = 0

            part_2 = np.maximum(np.abs(column_high - column_high.shift()),
                                np.abs(column_low - column_low.shift()))
            part_2[condition_2] = 0

            alpha = (part_1.rolling(sum_window).sum()
                     / (part_1.rolling(sum_window).sum()
                        + part_2.rolling(sum_window).sum()))
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_050(self):
        pass

    #--------------------------------------------------------------------------
    def alpha_051(self):
        pass

    #--------------------------------------------------------------------------
    def alpha_052(self, sum_window = 26):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()
            column_close = self.close[column].dropna()

            combine_lag_price = (
                (column_high + column_low + column_close) / 3).shift()
            part_1 = np.maximum(0, (column_high - combine_lag_price)).rolling(
                sum_window).sum()

            part_2 = np.maximum(0, (combine_lag_price - column_low)).rolling(
                sum_window).sum()

            alpha = part_1 / part_2
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_053(self, delay_window = 1, sum_window = 12):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            alpha = (column_close > column_close.shift(delay_window)).rolling(
                sum_window).sum()
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_054(self, liquid_contract_df,
                   std_window = 10, corr_window = 10):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_open = self.open[column].dropna()

            part_1 = (np.log(column_close)
                      - np.log(column_open)).abs().rolling(std_window).std()

            part_2 = np.log(column_close) - np.log(column_open)

            part_3 = column_open.rolling(corr_window).corr(column_close)

            alpha = part_1 + part_2 + part_3
            alpha_df[column] = alpha.reindex(self.index)

        alpha_liquid = self.get_liquid_contract_data(
            alpha_df, liquid_contract_df)
        alpha_rank = alpha_liquid.rank(axis = 1, pct = True)

        return alpha_rank

    #--------------------------------------------------------------------------
    def alpha_055(self):
        pass

    #--------------------------------------------------------------------------
    def alpha_056(self, liquid_contract_df, ts_min_window = 12,
                   sum_window = 19, volume_mean_window = 40,
                   corr_window = 13, time_num = 5):
        part_1_df = pd.DataFrame(index = self.index)
        part_2_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_open = self.open[column].dropna()
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()
            column_volume = self.volume[column].dropna()

            part_1 = column_open - column_open.rolling(ts_min_window).min()
            part_1_df[column] = part_1.reindex(self.index)

            tmp_1 = ((column_high+column_low)*0.5).rolling(sum_window).sum()
            tmp_2 = (column_volume.rolling(volume_mean_window).mean()).rolling(
                sum_window).sum()
            part_2 = tmp_1.rolling(corr_window).corr(tmp_2)
            part_2_df[column] = part_2.reindex(self.index)

        part_1_liquid = self.get_liquid_contract_data(
            part_1_df, liquid_contract_df)
        part_2_liquid = self.get_liquid_contract_data(
            part_2_df, liquid_contract_df)

        part_1_rank = part_1_liquid.rank(axis = 1, pct = True)
        part_2_rank = part_2_liquid.rank(axis = 1, pct = True)
        part_2_final_rank = (part_2_rank ** time_num).rank(
            axis = 1, pct = True)

        alpha_df = 0.0 * part_1_rank
        alpha_df[part_1_rank < part_2_final_rank] = 1.0

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_057(self, ts_window = 9, alpha = 1.0/3.0):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()

            part_1 = column_close - column_low.rolling(ts_window).min()
            part_2 = (column_high.rolling(ts_window).max()
                      - column_low.rolling(ts_window).min())
            alpha = (part_1/part_2*100).ewm(alpha = alpha).mean()
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_058(self, delay_window = 1, count_window = 20):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()

            alpha = (
                (column_close - column_close.shift(delay_window)) > 0).rolling(
                    count_window).sum()
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_059(self, delay_window = 1, sum_window = 20):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()

            delay_close = column_close.shift(delay_window)
            condition_1 = (column_close > delay_close)
            condition_2 = (column_close < delay_close)

            part_1 = np.minimum(
                column_low[condition_1],
                delay_close[condition_1]).fillna(0)
            part_2 = np.maximum(
                column_high[condition_2],
                delay_close[condition_2]).fillna(0)

            alpha = (column_close - part_1 - part_2).rolling(sum_window).sum()
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df

    #--------------------------------------------------------------------------
    def alpha_060(self, sum_window = 20):
        alpha_df = pd.DataFrame(index = self.index)

        for column in self.columns:
            column_close = self.close[column].dropna()
            column_high = self.high[column].dropna()
            column_low = self.low[column].dropna()
            column_volume = self.volume[column].dropna()

            alpha = ((((column_close-column_low)-(column_high-column_close))
                      /(column_high-column_low))*column_volume).rolling(
                sum_window).sum()
            alpha_df[column] = alpha.reindex(self.index)

        return alpha_df