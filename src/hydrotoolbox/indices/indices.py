# Standard library imports

# Third party imports
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# First party imports
from hydrotoolbox.toolbox_utils.src.toolbox_utils import tsutils


class Indices:
    def __init__(self, data, use_median=False, water_year="YE-SEP", drainage_area=1):
        if isinstance(data, pd.DataFrame) and len(data.columns) != 1:
            raise ValueError(
                tsutils.error_wrapper(f"""
                    Can only calculate indices on 1 series, you gave
                    {len(data.columns)}.
                    """)
            )

        self.use_median = use_median
        self.water_year = tsutils.pandas_offset_by_version(water_year)
        self.drainage_area = float(drainage_area)

        self.data = pd.Series(data.iloc[:, 0].values, index=data.index)
        self.data[self.data < 0] = pd.NA
        self.data = self.data.dropna()

        self.data_monthly = self.data.groupby(
            pd.Grouper(freq=tsutils.pandas_offset_by_version("ME"))
        )
        self.data_yearly = self.data.groupby(pd.Grouper(freq=self.water_year))

        self.data_monthly_mean = self.data_monthly.mean()
        self.data_yearly_mean = self.data_yearly.mean()

        self.data_monthly_min = self.data_monthly.min()
        self.data_yearly_min = self.data_yearly.min()

        self.data_monthly_max = self.data_monthly.max()
        self.data_yearly_max = self.data_yearly.max()

    def MA1(self):
        """MA1

        Mean of the daily mean flow values for the entire flow record.
        cubic feet per second—temporal"""
        return self.data.mean()

    def MA2(self):
        """MA2

        Median of the daily mean flow values for the entire flow record.
        cubic feet per second—temporal"""
        return self.data.median()

    def MA3(self):
        """MA3

        Mean (or median) of the coefficients of variation (standard
        deviation/mean) for each year.  Compute the coefficient of variation
        for each year of daily flows. Compute the mean of the annual
        coefficients of variation.
        percent—temporal"""
        tmpdata = self.data_yearly.std() / self.data_yearly.mean()
        return tmpdata.median() * 100 if self.use_median else tmpdata.mean() * 100

    def MA4(self):
        """MA4

        Standard deviation of the percentiles of the logs of the entire flow
        record divided by the mean of percentiles of the logs.   Compute the
        log10 of the daily flows for the entire record.  Compute the 5th, 10th,
        15th, 20th, 25th, 30th, 35th, 40th, 45th, 50th, 55th, 60th, 65th, 70th,
        75th, 80th, 85th, 90th, and 95th  percentiles for the logs of the
        entire flow record.

        Percentiles are computed by interpolating between the ordered
        (ascending) logs of the flow values. Compute the standard deviation
        and mean for the percentile values. Divide the standard deviation by
        the mean.
        percent-spatial"""
        p = [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ]
        newp = self.data.quantile(p)
        return newp.std() / newp.mean() * 100

    def MA5(self):
        """MA5

        The skewness of the entire flow record is computed as the mean for the
        entire flow record (MA1) divided by the median (MA2) for the entire
        flow record.
        dimensionless—spatial"""
        return self.MA1() / self.MA2()

    def _MA_6_8(self, high: float, low: float):
        return self.data.quantile(high) / self.data.quantile(low)

    def MA6(self):
        """MA6

        Range in daily flows is the ratio of the 10-percent to 90-percent  exceedance
        values for the entire flow record. Compute the 5-percent to 95-percent
        exceedance values for the entire flow record. Exceedance is computed by
        interpolating between the   ordered (descending) flow values.  Divide the
        10-percent exceedance value by the 90-percent value.
        dimensionless—spatial
        """
        return self._MA_6_8(0.9, 0.1)

    def MA7(self):
        """MA7

        Range in daily flows is computed like MA6, except using the 20 percent and 80
        percent exceedance values. Divide the 20 percent exceedance value by the 80
        percent value.
        dimensionless—spatial
        """
        return self._MA_6_8(0.8, 0.2)

    def MA8(self):
        """MA8

        Range in daily flows is computed like MA6, except using the 25-percent and
        75-percent exceedance values. Divide the 25-percent exceedance value by the
        75-percent value.
        dimensionless—spatial
        """
        return self._MA_6_8(0.75, 0.25)

    def _MA_9_11(self, high: float, low: float):
        return (self.data.quantile(high) - self.data.quantile(low)) / self.MA2()

    def MA9(self):
        """MA9

        Spread in daily flows is the ratio of the difference between the 90th and 10th
        percentile of the logs of the flow data to the log of the median of the entire
        flow record. Compute the log10 of the daily flows for the entire record.
        Compute the 5th, 10th, 15th, 20th, 25th, 30th, 35th, 40th, 45th, 50th, 55th,
        60th, 65th, 70th, 75th, 80th, 85th, 90th, and 95th percentiles for the logs of
        the entire flow record. Percentiles are computed by interpolating between the
        ordered (ascending) logs of the flow values.  Compute MA9 as (90th –10th)
        /log10(MA2).
        dimensionless—spatial
        """
        return self._MA_9_11(0.9, 0.1)

    def MA10(self):
        """MA10

        Spread in daily flows is computed like MA9, except using the 20th and 80th
        percentiles.
        dimensionless—spatial
        """
        return self._MA_9_11(0.8, 0.2)

    def MA11(self):
        """MA11

        Spread in daily flows is computed like MA9, except using the 25th and 75th
        percentiles.
        dimensionless—spatial
        """
        return self._MA_9_11(0.75, 0.25)

    def _MA_12_23(self, month: int):
        if self.use_median is True:
            return self.data[self.data.index.month == month].median()
        return self.data[self.data.index.month == month].mean()

    def MA12(self):
        """MA12

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA12 is the mean of all January flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(1)

    def MA13(self):
        """MA13

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA13 is the mean of all February flow values over the entire record
        MA12 is the mean of all January flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(2)

    def MA14(self):
        """MA14

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA14 is the mean of all March flow values over the entire record
        MA12 is the mean of all January flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(3)

    def MA15(self):
        """MA15

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA15 is the mean of all April flow values over the entire record
        MA12 is the mean of all January flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(4)

    def MA16(self):
        """MA16

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA16 is the mean of all May flow values over the entire record
        MA12 is the mean of all January flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(5)

    def MA17(self):
        """MA17

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA17 is the mean of all June flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(6)

    def MA18(self):
        """MA18

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA18 is the mean of all July flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(7)

    def MA19(self):
        """MA19

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA19 is the mean of all August flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(8)

    def MA20(self):
        """MA20

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA20 is the mean of all September flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(9)

    def MA21(self):
        """MA21

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA21 is the mean of all October flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(10)

    def MA22(self):
        """MA22

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA22 is the mean of all November flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(11)

    def MA23(self):
        """MA23

        Means (or medians) of monthly flow values. Compute the means for each.  Means
        (or medians) of monthly flow values. Compute the means for each month over the
        entire flow record.

        MA23 is the mean of all December flow values over the entire record
        (cubic feet per second— temporal).
        """
        return self._MA_12_23(12)

    def _MA_24_35(self, month: int):
        group = self.data.groupby(
            pd.Grouper(freq=tsutils.pandas_offset_by_version("ME"))
        )
        tmpdata = group.std() / group.mean()
        if self.use_median is True:
            return tmpdata[tmpdata.index.month == month].median() * 100
        return tmpdata[tmpdata.index.month == month].mean() * 100

    def MA24(self):
        """MA24

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA24 is the variability of all January flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(1)

    def MA25(self):
        """MA25

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA25 is the variability of all February flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(2)

    def MA26(self):
        """MA26

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA26 is the variability of all March flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(3)

    def MA27(self):
        """MA27

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA27 is the variability of all April flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(4)

    def MA28(self):
        """MA28

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA28 is the variability of all May flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(5)

    def MA29(self):
        """MA29

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA29 is the variability of all June flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(6)

    def MA30(self):
        """MA30

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA30 is the variability of all July flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(7)

    def MA31(self):
        """MA31

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA31 is the variability of all August flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(8)

    def MA32(self):
        """MA32

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA32 is the variability of all September flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(9)

    def MA33(self):
        """MA33

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA33 is the variability of all October flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(10)

    def MA34(self):
        """MA34

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA34 is the variability of all November flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(11)

    def MA35(self):
        """MA35

        Variability (coefficient of variation) of monthly flow values. Compute the
        standard deviation for each.  Variability (coefficient of month in each year
        over the entire flow record. Divide the standard deviation by the mean for each
        month. Average (or take median of) these values for each month across all
        years.

        MA35 is the variability of all December flow values over the entire record.
        percent—temporal
        """
        return self._MA_24_35(12)

    def MA36(self):
        """MA36

        Variability across monthly flows. Compute the minimum, maximum, and
        mean flows for each month in the entire flow record.  MA36 is the
        maximum monthly flow minus the minimum monthly flow divided by the
        median monthly flow.
        dimensionless-spatial"""
        return (
            max(self.data_monthly_mean) - min(self.data_monthly_mean)
        ) / self.data_monthly_mean.median()

    def _MA_37_38(self, high: float, low: float):
        return (
            self.data_monthly_mean.quantile(high) - self.data_monthly_mean.quantile(low)
        ) / self.data_monthly_mean.median()

    def MA37(self):
        """MA37

        Variability across monthly flows. Compute the first (25th percentile) and the
        third (75th percentile) quartiles (every month in dimensionless— the flow
        record). MA37 is the third quartile minus the first quartile divided by the
        median of the monthly means.
        dimensionless-spatial
        """
        return self._MA_37_38(0.75, 0.25)

    def MA38(self):
        """MA38

        Variability across monthly flows. Compute the 10th and 90th percentiles for the
        monthly means (every month in the flow record). MA38 is the 90th percentile
        minus the 10th percentile divided by the median of the monthly means.
        dimensionless—spatial
        """
        return self._MA_37_38(0.9, 0.1)

    def MA39(self):
        """MA39

        Variability across monthly flows. Compute the standard deviation for
        the monthly means. MA39 is the standard deviation times 100 divided by
        the mean of the monthly means.
        percent—spatial"""
        return self.data_monthly_mean.std() / self.data_monthly_mean.mean() * 100

    def MA40(self):
        """MA40

        Skewness in the monthly flows. MA40 is the mean of the monthly flow
        means minus the median of the monthly means divided by the median of
        the monthly means.
        dimensionles-sspatial"""
        return (
            self.data_monthly_mean.mean() - self.data_monthly_mean.median()
        ) / self.data_monthly_mean.median()

    def MA41(self):
        """MA41

        Annual runoff. Compute the annual mean daily flows. MA41 is the mean of
        the annual means divided by the drainage area.
        cubic feet per second/ square mile—temporal"""
        return self.data_yearly_mean.mean() / self.drainage_area

    def MA42(self):
        """MA42

        Variability across annual flows. MA42 is the maximum annual flow minus
        the minimum annual flow divided by the median annual flow.
        dimensionless-spatial"""
        return (
            max(self.data_yearly_mean) - min(self.data_yearly_mean)
        ) / self.data_yearly_mean.median()

    def _MA_43_44(self, high: float, low: float):
        return (
            self.data_yearly_mean.quantile(high) - self.data_yearly_mean.quantile(low)
        ) / self.data_yearly_mean.median()

    def MA43(self):
        """MA43

        Variability across annual flows. Compute the first (25th percentile) and third
        (75th percentile) quartiles for the annual means (every year in the flow
        record).

        MA43 is the third quartile minus the first quartile divided by the median of
        the annual means.
        dimensionless-spatial
        """
        return self._MA_43_44(0.75, 0.25)

    def MA44(self):
        """MA44

        Variability across annual flows. Compute the 10th and 90th percentiles for the
        annual means (every year in the flow record).

        MA44 is the 90th percentile minus the 10th percentile divided by the median of
        the annual means.
        dimensionless-spatial
        """
        return self._MA_43_44(0.9, 0.1)

    def MA45(self):
        """MA45

        Skewness in the annual flows. MA45 is the mean of the annual flow means
        minus the median of the annual means divided by the median of the
        annual means.
        dimensionless-spatial"""
        return (
            self.data_yearly_mean.mean() - self.data_yearly_mean.median()
        ) / self.data_yearly_mean.median()

    def _ML_1_12(self, month: int):
        if self.use_median is True:
            return self.data_monthly_min[
                self.data_monthly_min.index.month == month
            ].median()
        return self.data_monthly_min[self.data_monthly_min.index.month == month].mean()

    def ML1(self):
        """ML1

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML1 is the mean of the minimums of all January flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(1)

    def ML2(self):
        """ML2

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML2 is the mean of the minimums of all February flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(2)

    def ML3(self):
        """ML3

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML3 is the mean of the minimums of all March flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(3)

    def ML4(self):
        """ML4

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML4 is the mean of the minimums of all April flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(4)

    def ML5(self):
        """ML5

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML5 is the mean of the minimums of all May flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(5)

    def ML6(self):
        """ML6

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML6 is the mean of the minimums of all June flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(6)

    def ML7(self):
        """ML7

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML7 is the mean of the minimums of all July flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(7)

    def ML8(self):
        """ML8

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML8 is the mean of the minimums of all August flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(8)

    def ML9(self):
        """ML9

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML9 is the mean of the minimums of all September flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(9)

    def ML10(self):
        """ML10

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML10 is the mean of the minimums of all October flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(10)

    def ML11(self):
        """ML11

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML11 is the mean of the minimums of all November flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(11)

    def ML12(self):
        """ML12

        Mean (or median) of minimum flows for each month across all years. Compute the
        minimums for each month over the entire flow record.

        ML12 is the mean of the minimums of all December flow values over the entire
        record.
        cubic feet per second—temporal
        """
        return self._ML_1_12(12)

    def ML13(self):
        """ML13

        Variability (coefficient of variation) across minimum monthly flow
        values. Compute the mean and standard deviation for the minimum monthly
        flows over the entire flow record. ML13 is the standard deviation times
        100 divided by the mean minimum monthly flow for all years.
        percent—spatial"""
        return self.data_monthly_min.std() / self.data_monthly_min.mean() * 100

    def ML14(self):
        """ML14

        Compute the minimum annual flow for each year. ML14 is the mean of the
        ratios of minimum annual flows to the median flow for each year.
        dimensionless—temporal"""
        return (
            self.data_yearly_min
            / self.data.groupby(pd.Grouper(freq=self.water_year)).median()
        ).mean()

    def ML15(self):
        """ML15

        Low-flow index. ML15 is the mean of the ratios of minimum annual flows
        to the mean flow for each year.
        dimensionless—temporal"""
        return (self.data_yearly_min / self.data_yearly_mean).mean()

    def ML16(self):
        """ML16

        Median of annual minimum flows. ML16 is the median of the ratios of
        minimum annual flows to the median flow for each year.
        dimensionless— temporal"""
        return (self.data_yearly_min / self.data_yearly.median()).median()

    def ML17(self):
        """ML17

        Base flow. Compute the mean annual flows. Compute the minimum of
        a 7-day moving average flows for each year and divide them by the mean
        annual flow for that year. ML17 is the mean (or median if use_median is
        set) of those ratios.
        dimensionless—temporal"""
        stat = (
            self.data.rolling(7).mean().groupby(pd.Grouper(freq=self.water_year)).min()
            / self.data_yearly_mean
        )
        return stat.median() if self.use_median is True else stat.mean()

    def ML18(self):
        """ML18

        Variability in base flow. Compute the standard deviation for the ratios
        of 7-day moving average flows to mean annual flows for each year. ML18
        is the standard deviation times 100 divided by the mean of the ratios.
        percent—spatial"""
        ratios = (
            self.data.rolling(7).mean().groupby(pd.Grouper(freq=self.water_year))
        ).min() / self.data_yearly_mean
        return ratios.std() / ratios.mean() * 100

    def ML19(self):
        """ML19

        Base flow. Compute the ratios of the minimum annual flow to mean annual
        flow for each year. ML19 is the mean (or median) of these ratios times
        100.
        dimensionless—temporal"""
        if self.use_median is True:
            return (self.data_yearly_min / self.data_yearly_mean).median() * 100
        return (self.data_yearly_min / self.data_yearly_mean).mean() * 100

    def ML20(self):
        """ML20

        Base flow. Divide the daily flow record into 5-day blocks. Find the
        minimum flow for each block. Assign the minimum flow as a base flow for
        that block if 90 percent of that minimum flow is less than the minimum
        flows for the blocks on either side. Otherwise, set it to zero. Fill in
        the zero values using linear interpolation. Compute the total flow for
        the entire record and the total base flow for the entire record. ML20
        is the ratio of total flow to total base flow.
        dimensionless—spatial"""
        # Local folder imports
        from ..baseflow_sep import five_day

        newq = five_day(self.data)
        return (newq.sum() / self.data.sum()).iloc[0]

    def ML21(self):
        """ML21

        Variability across annual minimum flows. Compute the mean and standard
        deviation for the annual minimum flows. ML21 is the standard deviation
        times 100 divided by the mean.
        percent—spatial"""
        return self.data_yearly_min.std() / self.data_yearly_min.mean() * 100

    def ML22(self):
        """ML22

        Specific mean annual minimum flow. ML22 is the mean (or median) of the
        annual minimum flows divided by the drainage area.
        cubic feet per second/square mile—temporal"""
        return self.data_yearly_min.mean() / self.drainage_area

    def _MH_1_12(self, month: int):
        stat = self.data_monthly_max[self.data_monthly_max.index.month == month]
        return stat.median() if self.use_median is True else stat.mean()

    def MH1(self):
        """
        MH1: Mean (or median) maximum flows for each month across all years.

        Compute the maximums for each month over the entire cubic feet per flow
        record.

        MH1 is the mean of the maximums of all January flow values over the
        entire record.
        second—temporal
        """
        return self._MH_1_12(1)

    def MH2(self):
        """MH2

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH2 is the mean of the maximums of all February flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(2)

    def MH3(self):
        """MH3

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH3 is the mean of the maximums of all March flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(3)

    def MH4(self):
        """MH4

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH4 is the mean of the maximums of all April flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(4)

    def MH5(self):
        """MH5

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH5 is the mean of the maximums of all May flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(5)

    def MH6(self):
        """MH6

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH6 is the mean of the maximums of all June flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(6)

    def MH7(self):
        """MH7

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH7 is the mean of the maximums of all July flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(7)

    def MH8(self):
        """MH8

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH8 is the mean of the maximums of all August flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(8)

    def MH9(self):
        """MH9

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH9 is the mean of the maximums of all September flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(9)

    def MH10(self):
        """MH10

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH10 is the mean of the maximums of all October flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(10)

    def MH11(self):
        """MH11

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH11 is the mean of the maximums of all November flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(11)

    def MH12(self):
        """MH12

        Mean (or median) maximum flows for each month across all years. Compute the
        maximums for each month over the entire cubic feet per flow record.

        MH12 is the mean of the maximums of all December flow values over the entire
        record.
        second—temporal
        """
        return self._MH_1_12(12)

    def MH13(self):
        """MH13

        Variability (coefficient of variation) across maximum monthly flow
        values. Compute the mean and standard deviation for the maximum monthly
        flows over the entire flow record. MH13 is the standard deviation times
        100 divided by the mean maximum monthly flow for all years.
        percent—spatial"""
        return self.data_monthly_max.std() / self.data_monthly_max.mean() * 100

    def MH14(self):
        """MH14

        Median of annual maximum flows. Compute the annual maximum flows from
        monthly maximum flows. Compute the ratio of annual maximum flow to
        median annual flow for each year. MH14 is the median of these ratios.
        dimensionless—temporal"""
        return (self.data_yearly_max / self.data_yearly.median()).median()

    def _MH_15_17(self, quant: float):
        return self.data.quantile(quant) / self.MA2()

    def MH15(self):
        """
        MH15

        High flow discharge index. Compute the 1-percent exceedance value for
        the entire data record.

        MH15 is the 1-percent exceedance value divided by the median flow for
        the entire record.
        dimensionless—spatial
        """
        return self.data.quantile(0.99) / self.MA2()

    def MH16(self):
        """MH16

        High flow discharge index. Compute the 1-percent exceedance value for the
        entire data record.

        MH16 is the 10-percent exceedance value divided by the median flow for the
        entire record.
        dimensionless—spatial
        """
        return self._MH_15_17(0.90)

    def MH17(self):
        """MH17

        High flow discharge index. Compute the 1-percent exceedance value for the
        entire data record.

        MH17 is the 25-percent exceedance value divided by the median flow for the
        entire record.
        dimensionless—spatial
        """
        return self._MH_15_17(0.75)

    def MH18(self):
        """MH18

        Variability across annual maximum flows. Compute the logs (log10) of
        the maximum annual flows. Find the standard percent—spatial deviation
        and mean for these values. MH18 is the standard deviation times 100
        divided by the mean."""
        log10 = np.log10(self.data_yearly_max)
        return log10.std() / log10.mean() * 100

    def MH19(self):
        """MH19

        Skewness in annual maximum flows.
        dimensionless—spatial
        """
        qm = np.log10(self.data_yearly_max)
        N = self.data_yearly_max.count()
        S = qm.std()
        return (
            N**2 * (qm**3).sum()
            - 3 * N * qm.sum() * (qm**2).sum()
            + 2 * (qm.sum()) ** 3
        ) / (N * (N - 1) * (N - 2) * S**3)

    def MH20(self):
        """MH20

        Specific mean annual maximum flow. MH20 is the mean (or median) of the
        annual maximum flows divided by the drainage area.
        cubic feet per second/square mile—temporal
        """
        if self.use_median:
            return self.data_yearly_max.median() / self.drainage_area
        return self.data_yearly_max.mean() / self.drainage_area

    def _MH_21_23(self, med_mult: int):
        med = self.MA2()
        flow = self.data - (med_mult * med)

        flow[flow < 0] = 0.0
        if flow[flow > 0].count() == 0:
            return None

        # Mean of the yearly volume (sum of flows) above the median
        # * med_mult
        qmean = flow.groupby(pd.Grouper(freq=self.water_year)).sum().mean()

        nevents = (flow > 0) & (flow.shift(-1) == 0)
        nevents = nevents.groupby(pd.Grouper(freq=self.water_year)).sum().mean()

        return qmean / nevents / med

    def MH21(self):
        """MH21

        High flow volume index. Compute the average volume for flow events above
        a threshold equal to the median flow for the entire record. MH21 is the
        average volume divided by the median flow for the entire record.
        days—temporal
        """
        return self._MH_21_23(1)

    def MH22(self):
        """MH22

        High flow volume. Compute the average volume for flow events above a
        threshold equal to three times the median flow for the entire record.
        MH22 is the average volume divided by the median flow for the entire
        record.
        days—temporal"""
        return self._MH_21_23(3)

    def MH23(self):
        """MH23

        High flow volume. Compute the average volume for flow events above a
        threshold equal to seven times the median flow for the entire record.
        MH23 is the average volume divided by the median flow for the entire
        record.
        days—temporal"""
        return self._MH_21_23(7)

    def _MH_24_26(self, med_mult: int = 1, quantile: float | None = None):
        if quantile is not None:
            medm = self.data.quantile(quantile)
        else:
            medm = self.MA2() * med_mult

        peaks, _ = find_peaks(self.data, height=medm)
        return self.data.iloc[peaks].mean() / self.MA2()

    def MH24(self):
        """MH24

        High peak flow. Compute the average peak flow value for flow events
        above a threshold equal to the median flow for the entire record.  MH24
        is the average peak flow divided by the median flow for the entire
        record.
        dimensionless—temporal
        """
        return self._MH_24_26(med_mult=1)

    def MH25(self):
        """MH25

        High peak flow.  Compute the average peak flow value for flow events
        above a threshold equal to three times the median flow for the entire
        record.  MH25 is the average peak flow divided by the median flow for
        the entire record.
        dimensionless—temporal
        """
        return self._MH_24_26(med_mult=3)

    def MH26(self):
        """MH26

        High peak flow. Compute the average peak flow value for flow events
        above a threshold equal to seven times the median flow for the entire
        record. MH26 is the average peak flow divided by the median flow for
        the entire record.
        dimensionless—temporal
        """
        return self._MH_24_26(med_mult=7)

    def MH27(self):
        """MH27

        High peak flow. Compute the average peak flow value for flow events
        above a threshold equal to the 75th percentile flow for the entire
        record. MH27 is the average peak flow divided by the median flow for
        the entire record.
        dimensionless—temporal
        """
        return self._MH_24_26(quantile=0.75)

    def event_statistics(self, thresh, than):
        """Event statistics.

        Parameters
        ----------
        thresh : float
            Threshold value.
        than : str
            Comparison operator.

        Returns
        -------
        event_counts : pandas.Series
            Number of matching events per year.
        average_durations : pandas.Series
            Average event duration for years with matching events.
        matching_days : pandas.Series
            Count of matching days per year.
        """
        nnp = {}
        lfdur = {}
        allnp = {}
        for group_name, group_df in self.data_yearly:
            nnp[group_name] = 0
            lfdur[group_name] = 0
            allnp[group_name] = 0
            pdur = 0
            flag = 0
            for index, value in group_df.items():
                if eval(f"value {than} thresh"):
                    pdur = pdur + 1
                    flag = flag + 1
                    allnp[group_name] += 1
                    if flag == 1:
                        nnp[group_name] += 1
                else:
                    flag = 0
            if nnp[group_name] > 0:
                lfdur[group_name] = pdur / nnp[group_name]
        lfdur = pd.Series(data=lfdur.values())
        lfdur = lfdur[lfdur > 0]
        return pd.Series(data=nnp.values()), lfdur, pd.Series(data=allnp.values())

    def FL1(self):
        """FL1

        Low flood pulse count. Compute the average number of flow events
        with flows below a threshold equal to the 25th-percentile value
        for the entire flow record. FL1 is the average (or median)
        number of events.
        number of events/year—temporal
        """
        thresh = self.data.quantile(0.25)
        nnp, _, _ = self.event_statistics(thresh, "<")
        return nnp.median() if self.use_median is True else nnp.mean()

    def FL2(self):
        """FL2

        Variability in low pulse count. Compute the standard deviation in
        the annual pulse counts for FL1. FL2 is 100 times the standard
        deviation divided by the mean pulse count.
        percent—spatial
        """
        thresh = self.data.quantile(0.25)
        nnp, _, _ = self.event_statistics(thresh, "<")
        return nnp.std() / nnp.mean() * 100

    def FL3(self):
        """FL3

        Frequency of low pulse spells. Compute the average number of flow
        events with flows below a threshold equal to 5 percent of the mean
        flow value for the entire flow record. FL3 is the average (or
        median) number of events.
        number of events/year—temporal
        """
        thresh = self.data.mean() * 0.05
        nnp, _, _ = self.event_statistics(thresh, "<")
        return nnp.median() if self.use_median is True else nnp.mean()

    def FH1(self):
        """FH1

        High flood pulse count. Compute the average number of flow events
        with flows above a threshold equal to the 75th-percentile value for
        the entire flow record. FH1 is the average (or median) number of
        events.
        number of events/year—temporal
        """
        thresh = self.data.quantile(0.75)
        nnp, _, _ = self.event_statistics(thresh, ">")
        return nnp.median() if self.use_median is True else nnp.mean()

    def FH2(self):
        """FH2

        Variability in high pulse count. Compute the standard deviation in
        the annual pulse counts for FH1. FH2 is 100 times the standard
        deviation divided by the mean pulse count.
        number of events/year—spatial
        """
        thresh = self.data.quantile(0.75)
        nnp, _, _ = self.event_statistics(thresh, ">")
        return nnp.std() / nnp.mean() * 100

    def _FH_3_4(self, med_mult: int):
        thresh = med_mult * self.MA2()
        _, _, pdur = self.event_statistics(thresh, ">")
        return pdur.median() if self.use_median is True else pdur.mean()

    def FH3(self):
        """FH3

        High flood pulse count. Compute the average number of days per year
        that the flow is above a threshold equal to three times the median
        flow for the entire record. FH3 is the mean (or median) of the
        annual number of days for all years.
        number of days/year—temporal
        """
        return self._FH_3_4(3)

    def FH4(self):
        """FH4

        High flood pulse count. Compute the average number of days per year
        that the flow is above a threshold equal to seven times the median
        flow for the entire record. FH4 is the mean (or median) of the
        annual number of days for all years.
        number of days/year—temporal
        """
        return self._FH_3_4(7)

    def _FH_5_7(self, med_mult: int):
        thresh = self.MA2() * med_mult
        nnp, _, _ = self.event_statistics(thresh, ">")
        return nnp.median() if self.use_median is True else nnp.mean()

    def FH5(self):
        """FH5

        Flood frequency. Compute the average number of flow events with
        flows above a threshold equal to the median flow value for the
        entire flow record. FH5 is the average (or median) number of events.
        number of events/year—temporal
        """
        return self._FH_5_7(1)

    def FH6(self):
        """FH6

        Flood frequency. Compute the average number of flow events with
        flows above a threshold equal to three times the median flow value
        for the entire flow record. FH6 is the average (or median) number of
        events.
        number of events/year—temporal
        """
        return self._FH_5_7(3)

    def FH7(self):
        """FH7

        Flood frequency. Compute the average number of flow events with
        flows above a threshold equal to seven times the median flow value
        for the entire flow record. FH7 is the average (or median) number of
        events.
        number of events/year—temporal
        """
        return self._FH_5_7(7)

    def _FH_8_9(self, quant: float):
        thresh = self.data.quantile(quant)
        nnp, _, _ = self.event_statistics(thresh, ">")
        return nnp.median() if self.use_median is True else nnp.mean()

    def FH8(self):
        """FH8

        Flood frequency. Compute the average number of flow events with
        flows above a threshold equal to 25-percent exceedance value for the
        entire flow record. FH8 is the average (or median) number of events.
        number of events/year—temporal
        """
        return self._FH_8_9(0.75)

    def FH9(self):
        """FH9

        Flood frequency. Compute the average number of flow events with
        flows above a threshold equal to 75-percent exceedance value for the
        entire flow record. FH9 is the average (or median) number of events.
        number of events/year—temporal
        """
        return self._FH_8_9(0.25)

    def FH10(self):
        """FH10

        Flood frequency. Compute the average number of flow events with
        flows above a threshold equal to median of the annual cubic
        feet/second minima for the entire flow record. FH10 is the average
        (or median) number of events.
        number of events/year—temporal
        """
        thresh = self.data_yearly.min().median()
        nnp, _, _ = self.event_statistics(thresh, ">")
        return nnp.median() if self.use_median is True else nnp.mean()

    def _wy_start_month(self):
        return pd.tseries.frequencies.to_offset(self.water_year).month % 12 + 1

    def _wy_start(self):
        """Start date of the water year containing each data point."""
        start_month = self._wy_start_month()
        idx = self.data.index
        year = idx.year - (idx.month < start_month).astype(int)
        return pd.Series(
            pd.to_datetime({"year": year, "month": start_month, "day": 1}).values,
            index=idx,
        )

    def _wy_day(self):
        """Zero based day of the water year for each data point."""
        return (self.data.index.normalize() - pd.DatetimeIndex(self._wy_start())).days

    def _wy_length(self):
        starts = pd.Series(self._wy_start().unique())
        return pd.Series(
            (starts + pd.DateOffset(years=1) - starts).dt.days.values,
            index=starts.values,
        )

    @staticmethod
    def _run_lengths(cond):
        """Lengths of consecutive True runs in a boolean Series."""
        rid = (cond != cond.shift()).cumsum()
        return cond[cond].groupby(rid[cond]).size()

    def _yearly_run_stat(self, thresh, stat):
        """Median or max length of runs at or below `thresh` in each water
        year, 0 for years without such runs."""

        def template(vals):
            runs = self._run_lengths(vals <= thresh)
            return getattr(runs, stat)() if len(runs) else 0

        return self.data_yearly.apply(template)

    def _seasonal_predictability(self, cond):
        """Maximum fraction of the days in `cond` that fall in one 2-month
        period of the water year."""
        period = ((self.data.index.month - self._wy_start_month()) % 12) // 2
        counts = cond.groupby(period).sum()
        total = counts.sum()
        return counts.max() / total if total > 0 else np.nan

    def _flood_threshold(self, return_period):
        """Flow for a given return period (years) of the annual peak flows.

        Only daily values are available, so the annual peak is the annual
        maximum daily flow, and the regression of daily flow on peak flow
        used by TSPROC reduces to the identity.  The quantile of the log10
        annual maxima is 1 - 1/return_period, using the Weibull plotting
        position (p = i/(n + 1)) appropriate for return periods.
        """
        peaks = self.data_yearly_max
        peaks = peaks[peaks > 0]
        return 10 ** np.quantile(
            np.log10(peaks), 1 - 1 / return_period, method="weibull"
        )

    def FH11(self):
        """
        Flood frequency. Compute the average number of flow events with flows
        above a threshold equal to flow corresponding to a 1.67-year recurrence
        interval (Poff, 1996; see index FH10 for computation details). FH11 is
        the average (or median) number of events.
        """
        thresh = self._flood_threshold(1.67)
        nnp, _, _ = self.event_statistics(thresh, ">")
        return nnp.median() if self.use_median is True else nnp.mean()

    def DL1(self):
        """DL1

        Annual minimum daily flow. Compute the minimum 1-day average flow
        for each year. DL1 is the mean (or median) of these values.
        cubic feet per second—temporal
        """
        stat = self.data_yearly.min()
        return stat.median() if self.use_median is True else stat.mean()

    def _preroll(self, days, stattype):
        stats = []
        for _, vals in self.data_yearly:
            rmean = vals.rolling(days).mean()
            if stattype == "min":
                stats.append(rmean.min())
            elif stattype == "max":
                stats.append(rmean.max())
        return pd.Series(data=stats, index=range(len(stats)))

    def _roll(self, days, stattype):
        stat = self._preroll(days, stattype)
        return stat.median() if self.use_median is True else stat.mean()

    def DL2(self):
        """DL2

        Annual minimum of 3-day moving average flow. Compute the minimum of
        a 3-day moving average flow for each year. DL2 is the mean (or
        median) of these values.
        cubic feet per second—temporal
        """
        return self._roll(3, "min")

    def DL3(self):
        """DL3

        Annual minimum of 7-day moving average flow. Compute the minimum of
        a 7-day moving average flow for each year. DL3 is the mean (or
        median) of these values.
        cubic feet per second—temporal
        """
        return self._roll(7, "min")

    def DL4(self):
        """DL4

        Annual minimum of 30-day moving average flow. Compute the minimum of
        a 30-day moving average flow for each year. DL4 is the mean (or
        median) of these values.
        cubic feet per second—temporal
        """
        return self._roll(30, "min")

    def DL5(self):
        """DL5

        Annual minimum of 90-day mean of flow
        """
        return self._roll(90, "min")

    def _DL_6_10_DH_6_10(self, days: int, instat):
        stat = self._preroll(days, instat)
        return stat.std() / stat.mean() * 100

    def DL6(self):
        """DL6

        CV: annual minimum of 1-day mean of flow
        """
        return self._DL_6_10_DH_6_10(1, "min")

    def DL7(self):
        """DL7

        CV: annual minimum of 3-day mean of flow
        """
        return self._DL_6_10_DH_6_10(3, "min")

    def DL8(self):
        """DL8

        CV: annual minimum of 7-day mean of flow
        """
        return self._DL_6_10_DH_6_10(7, "min")

    def DL9(self):
        """DL9

        CV: annual minimum of 30-day mean of flow
        """
        return self._DL_6_10_DH_6_10(30, "min")

    def DL10(self):
        """DL10

        CV: annual minimum of 90-day mean of flow
        """
        return self._DL_6_10_DH_6_10(90, "min")

    def DL11(self):
        """DL11

        Mean of 1-day minimum of flow
        """
        return self.data_yearly_min.mean() / self.MA2()

    def _DL_12_13_DH_12_13(self, days: int, instat: str):
        stat = self._preroll(days, instat)
        return stat.mean() / self.MA2()

    def DL12(self):
        """DL12

        Mean of 7-day minimum of flow
        """
        return self._DL_12_13_DH_12_13(7, "min")

    def DL13(self):
        """DL13

        Mean of 30-day minimum of flow
        """
        return self._DL_12_13_DH_12_13(30, "min")

    def DL14(self):
        """DL14

        Mean of Q75 values/median daily flow across all years
        """
        return self.data.quantile(0.25) / self.MA2()

    def DL15(self):
        """DL15

        Mean of Q90 values/median daily flow across all years
        """
        return self.data.quantile(0.1) / self.MA2()

    def DL16(self):
        """DL16

        Low flow pulse duration (Mean duration of FL1)
        """
        thresh = self.data.quantile(0.25)
        _, lfdur, _ = self.event_statistics(thresh, "<")
        return lfdur.median()

    def DL17(self):
        """DL17

        CV: low flow pulse duration (DL16)
        """
        thresh = self.data.quantile(0.25)
        _, lfdur, _ = self.event_statistics(thresh, "<")
        return lfdur.std() / lfdur.mean() * 100

    def DL18(self):
        """DL18

        Mean annual number of zero-flow days
        """
        stat = (
            self.data[self.data == 0].groupby(pd.Grouper(freq=self.water_year)).count()
        )
        if any(stat):
            return stat.median() if self.use_median is True else stat.mean()
        return 0

    def DL19(self):
        """DL19

        CV: mean annual number of zero-flow days
        """
        stat = (
            self.data[self.data == 0].groupby(pd.Grouper(freq=self.water_year)).count()
        )
        ret = stat.std() / stat.mean() * 100
        try:
            return ret.fillna(0)
        except AttributeError:
            return 0 if isinstance(ret, type(pd.NA)) else ret

    def DL20(self):
        """DL20

        Percentage of all months with zero flow
        """
        return self.data_monthly_mean[self.data_monthly_mean == 0].count()

    def DH1(self):
        """DH1

        Annual maximum of 1-day mean of flow
        """
        stat = self.data_yearly.max()
        stat = stat.median() if self.use_median else stat.mean()
        return stat

    def DH2(self):
        """DH2

        Annual maximum of 3-day mean of flow
        """
        return self._roll(3, "max")

    def DH3(self):
        """DH3

        Annual maximum of 7-day mean of flow
        """
        return self._roll(7, "max")

    def DH4(self):
        """DH4

        Annual maximum of 30-day mean of flow
        """
        return self._roll(30, "max")

    def DH5(self):
        """DH5

        Annual maximum of 90-day mean of flow
        """
        return self._roll(90, "max")

    def DH6(self):
        """DH6

        CV: annual maximum of 1-day mean of flow
        """
        return self._DL_6_10_DH_6_10(1, "max")

    def DH7(self):
        """DH7

        CV: annual maximum of 3-day mean of flow
        """
        return self._DL_6_10_DH_6_10(3, "max")

    def DH8(self):
        """DH8

        CV: annual maximum of 7-day mean of flow
        """
        return self._DL_6_10_DH_6_10(7, "max")

    def DH9(self):
        """DH9

        CV: annual maximum of 30-day mean of flow
        """
        return self._DL_6_10_DH_6_10(30, "max")

    def DH10(self):
        """DH10

        CV: annual maximum of 90-day mean of flow
        """
        return self._DL_6_10_DH_6_10(90, "max")

    def DH11(self):
        """DH11

        Mean of 1-day maximum of flow
        """
        return self.data_yearly_max.mean() / self.MA2()

    def DH12(self):
        """DH12

        Mean of 7-day maximum of flow
        """
        return self._DL_12_13_DH_12_13(7, "max")

    def DH13(self):
        """DH13

        Mean of 30-day maximum of flow
        """
        return self._DL_12_13_DH_12_13(30, "max")

    def DH14(self):
        """DH14

        Q95 value/mean monthly flow across all years
        """
        return self.data_monthly_mean.quantile(0.95) / self.data_monthly_mean.mean()

    def DH15(self):
        """DH15

        Mean duration of flood pulses>75th percentile flow
        """
        thresh = self.data.quantile(0.75)
        _, lfdur, _ = self.event_statistics(thresh, ">")
        return lfdur.median()

    def DH16(self):
        """DH16

        CV: mean duration of high flow pulse (FH1)
        """
        thresh = self.data.quantile(0.75)
        _, lfdur, _ = self.event_statistics(thresh, ">")
        return lfdur.std() / lfdur.mean() * 100

    def DH17(self):
        """DH17

        Mean duration of flood pulses>1*median flow
        """
        thresh = self.MA2()
        _, lfdur, _ = self.event_statistics(thresh, ">")
        return lfdur.median() if self.use_median else lfdur.mean()

    def DH18(self):
        """DH18

        Mean duration of flood pulses>3*median flow
        """
        thresh = self.MA2() * 3
        _, lfdur, _ = self.event_statistics(thresh, ">")
        return lfdur.median() if self.use_median else lfdur.mean()

    def DH19(self):
        """DH19

        Mean duration of flood pulses>7*median flow
        """
        thresh = self.MA2() * 7
        _, lfdur, _ = self.event_statistics(thresh, ">")
        return lfdur.median() if self.use_median else lfdur.mean()

    def DH20(self):
        """DH20

        Mean duration of flood pulses>75th percentile of median flow
        """
        thresh = self.data.quantile(0.75)
        _, lfdur, _ = self.event_statistics(thresh, ">")
        return lfdur.median() if self.use_median else lfdur.mean()

    def DH21(self):
        """DH21

        Mean duration of flood pulses>25th percentile of median flow
        """
        thresh = self.data.quantile(0.25)
        _, lfdur, _ = self.event_statistics(thresh, ">")
        return lfdur.median() if self.use_median else lfdur.mean()

    def DH22(self):
        """
        Flood interval. Compute the flood threshold as the flow equivalent for a
        flood recurrence of 1.67 years. Determine the median number of days
        between flood events for each year. DH22 is the mean (or median) of the
        yearly median number of days between flood events.
        """
        thresh = self._flood_threshold(1.67)
        stat = self._yearly_run_stat(thresh, "median")
        return stat.median() if self.use_median is True else stat.mean()

    def DH23(self):
        """
        Flood duration. Compute the flood threshold as the flow equivalent for a
        flood recurrence of 1.67 years. Determine the number of days each year
        that the flow remains above the flood threshold. DH23 is the mean (or
        median) of the number of flood days for years when floods occur.
        """
        thresh = self._flood_threshold(1.67)
        stat = self.data_yearly.apply(lambda vals: (vals > thresh).sum())
        stat = stat[stat > 0]
        return stat.median() if self.use_median is True else stat.mean()

    def DH24(self):
        """
        Flood-free days. Compute the flood threshold as the flow equivalent for
        a flood recurrence of 1.67 years. Compute the maximum number of days
        that the flow is below the threshold for each year. DH24 is the mean (or
        median) of the maximum yearly no-flood days.
        """
        thresh = self._flood_threshold(1.67)
        stat = self._yearly_run_stat(thresh, "max")
        return stat.median() if self.use_median is True else stat.mean()

    def _pre_ta1_ta2(self):
        nrows = 11
        lq = self.data.apply(np.log10)
        lma1 = np.log10(self.MA1())

        lq[self.data == 0.0] = np.log10(0.01)

        table = [
            lq[lq < 0.1 * lma1],
            lq[(lq >= 0.1 * lma1) & (lq < 0.25 * lma1)],
            lq[(lq >= 0.25 * lma1) & (lq < 0.5 * lma1)],
            lq[(lq >= 0.5 * lma1) & (lq < 0.75 * lma1)],
            lq[(lq >= 0.75 * lma1) & (lq < lma1)],
            lq[(lq >= lma1) & (lq < 1.25 * lma1)],
            lq[(lq >= 1.25 * lma1) & (lq < 1.5 * lma1)],
            lq[(lq >= 1.5 * lma1) & (lq < 1.75 * lma1)],
            lq[(lq >= 1.75 * lma1) & (lq < 2.0 * lma1)],
            lq[(lq >= 2.0 * lma1) & (lq < 2.25 * lma1)],
            lq[lq >= 2.25 * lma1],
        ]

        ndf = pd.DataFrame()
        for indx, df in enumerate(table):
            if df.empty:
                ndf = pd.concat(
                    [
                        ndf,
                        pd.DataFrame(
                            data=[0] * 365, index=range(1, 366), columns=[indx]
                        ),
                    ],
                    axis="columns",
                )
                continue
            ldf = [df[df.index.dayofyear == day].count() for day in range(1, 366)]
            ndf = pd.concat(
                [ndf, pd.DataFrame(data=ldf, index=range(1, 366), columns=[indx])],
                axis="columns",
            )
        Z = ndf.sum().sum()
        XJ = ndf.T.sum()
        YI = ndf.sum()
        HX = -(XJ / Z * np.log10(XJ / Z)).sum()
        HY = -(YI / Z * np.log10(YI / Z)).sum()
        HXY = -(ndf / Z * np.log10(ndf / Z)).sum().sum()
        HXY = HXY - HX
        return nrows, HY, HXY

    def TA1(self):
        """TA1

        Constancy (see Colwell: 1974)
        """
        nrows, HY, _ = self._pre_ta1_ta2()
        return 1 - (HY / np.log10(nrows))

    def TA2(self):
        """TA2

        Predictability of flow
        """
        nrows, _, HXY = self._pre_ta1_ta2()
        return 100 * (1 - (HXY / np.log10(nrows)))

    def TA3(self):
        """
        Seasonal predictability of flooding. Divide years into 2-month periods
        (that is, Oct-Nov, Dec-Jan, and so forth). Count the number of flood
        days (flow events with flows > 1.67-year flood) in each period over the
        entire flow record. TA3 is the maximum number of flood days in any one
        period divided by the total number of flood days.
        """
        thresh = self._flood_threshold(1.67)
        return self._seasonal_predictability(self.data > thresh)

    def _min_max_doy(self, stat):
        if stat == "min":
            jd = (
                self.data.groupby(pd.Grouper(freq=self.water_year))
                .idxmin()
                .dt.dayofyear
            )
        if stat == "max":
            jd = (
                self.data.groupby(pd.Grouper(freq=self.water_year))
                .idxmax()
                .dt.dayofyear
            )
        mask = jd > 365.25
        jd.loc[mask] = jd.loc[mask] - 365.25
        jd = jd * 2 * np.pi / 365.25
        xbar = np.cos(jd).mean()
        ybar = np.sin(jd).mean()
        return xbar, ybar

    def TL1(self):
        """TL1

        Julian date of annual minimum. Determine the Julian date of the minimum
        flow for each water year. Transform the dates to relative values on
        a circular scale (radians or degrees). Compute the x and y components
        for each year, and average them across all years. Compute the mean
        angle as the arc tangent of y-mean divided by x-mean. Transform the
        resultant angle back to Julian date.
        Julian day—spatial"""
        xbar, ybar = self._min_max_doy("min")
        TL1 = np.arctan2(ybar, xbar) * 180.0 / np.pi
        if TL1 < 0.0:
            TL1 = TL1 + 360.0
        TL1 = TL1 * 365.25 / 360.0
        return TL1

    def TL2(self):
        """TL2

        Variability in Julian date of annual minima. Compute the coefficient of
        variation for the mean x and y components, and convert to a date.
        Julian day—spatial"""
        xbar, ybar = self._min_max_doy("min")
        temp = np.sqrt(xbar * xbar + ybar * ybar)
        temp = np.sqrt(2 * (1 - temp))
        return temp * 180 / np.pi / 360 * 365.25

    def TL3(self):
        """
        Seasonal predictability of low flow. Divide years into 2-month periods
        (that is, Oct-Nov, Dec-Jan, and so forth). Count the number of low flow
        events (flow events with flows <= 5 year flood threshold) in each period
        over the entire flow record. TL3 is the maximum number of low flow
        events in any one period divided by the total number of low flow events.
        """
        thresh = self._flood_threshold(5)
        return self._seasonal_predictability(self.data <= thresh)

    def TL4(self):
        """
        Seasonal predictability of non-low flow. Compute the number of days that
        flow is above the 5-year flood threshold as the ratio dimensionless— of
        number of days to 365 or 366 (leap year) for each year. TL4 is the
        maximum of the yearly ratios.
        """
        thresh = self._flood_threshold(5)
        days = (self.data > thresh).groupby(self._wy_start()).sum()
        return (days / self._wy_length()).max()

    def TH1(self):
        """TH1

        Julian date of annual maximum. Determine the Julian date of the maximum
        flow for each year. Transform the dates to relative values on a circular
        scale (radians or degrees). Compute the x and y components for each
        year, and average them across all years. Compute the mean angle as the
        arc tangent of y-mean divided by x-mean. Transform the resultant angle
        back to Julian date.  Julian day—spatial"""
        xbar, ybar = self._min_max_doy("max")
        TH1 = np.arctan2(ybar, xbar) * 180.0 / np.pi
        if TH1 < 0.0:
            TH1 = TH1 + 360.0
        TH1 = TH1 * 365.25 / 360.0
        return TH1

    def TH2(self):
        """TH2

        Variability in Julian date of annual maxima. Compute the coefficient of
        variation for the mean x and y components and convert to a date.
        Julian days—spatial"""
        xbar, ybar = self._min_max_doy("max")
        temp = np.sqrt(xbar * xbar + ybar * ybar)
        temp = np.sqrt(2 * (1 - temp))
        return temp * 180 / np.pi / 360 * 365.25

    def TH3(self):
        """
        Seasonal predictability of non-flooding. Computed as the maximum
        proportion of a 365-day year that the flow is less than the 1.67-year
        flood threshold and also occurs in all years. Accumulate non-flood days
        that span all years. TH3 is the maximum length of those flood-free
        periods divided by 365.

        Maximum proportion of the year (number of days/365) during which no
        floods have ever occurred over the period of record.
        """
        thresh = self._flood_threshold(1.67)
        flood = (self.data > thresh).groupby(self._wy_day()).any()
        flood = flood.reindex(range(366), fill_value=False)
        runs = self._run_lengths(~flood)
        return (runs.max() if len(runs) else 0) / 365

    def _rise_rate(self):
        delt = self.data.shift(1) - self.data
        return delt[delt < 0]

    def _fall_rate(self):
        delt = self.data.shift(1) - self.data
        return delt[delt > 0]

    def RA1(self):
        """RA1

        Mean of positive changes from one day to next (rise rate)
        """
        if self.use_median:
            return -self._rise_rate().median()
        return -self._rise_rate().mean()

    def RA2(self):
        """RA2

        CV: mean of positive changes from one day to next (rise rate)
        """
        rr = self._rise_rate()
        return -rr.std() / rr.mean() * 100

    def RA3(self):
        """RA3

        Mean of negative changes from one day to next (fall rate)
        """
        if self.use_median:
            return self._fall_rate().median()
        return self._fall_rate().mean()

    def RA4(self):
        """RA4

        CV: mean of negative changes from one day to next (fall rate)
        """
        rr = self._fall_rate()
        return rr.std() / rr.mean() * 100

    def RA5(self):
        """RA5

        Ratio of days that are higher than previous day
        """
        rr = self._rise_rate()
        return len(rr) / len(self.data)

    def RA6(self):
        """RA6

        Median of difference in log of flows over two consecutive days of
        rising flow
        """
        log = self.data.apply(np.log)
        log = log.shift(1) - log
        log = -log[log < 0]
        return log.median()

    def RA7(self):
        """RA7

        Median of difference in log of flows over two consecutive days of
        falling flow
        """
        log = self.data.apply(np.log)
        log = log.shift(1) - log
        log = log[log > 0]
        return log.median()

    def _changes(self):
        peaks = find_peaks(self.data)[0]
        valleys = find_peaks(-self.data)[0]
        changes = np.append(peaks, valleys)
        changes = pd.DataFrame(self.data.iloc[changes])
        df = pd.DataFrame(range(len(changes.index)), index=changes.index)
        df = df.groupby(pd.Grouper(freq=self.water_year)).count()
        return df

    def RA8(self):
        """RA8

        Number of flow reversals from one day to the next
        """
        df = self._changes()
        return df.median()[0] if self.use_median else df.mean()[0]

    def RA9(self):
        """RA9

        CV: number of flow reversals from one day to the next
        """
        df = self._changes()
        return (df.std() / df.mean())[0] * 100


if __name__ == "__main__":
    df = pd.read_csv(
        # "../../../tests/data_02239501.csv", index_col="Datetime", parse_dates=True
        "../../../tests/data_percentile_test.csv",
        index_col="Datetime",
        parse_dates=True,
    )
    ndf = df.iloc[:, 0]
    ind = Indices(ndf)
    print("MA1 = ", ind.MA1())
    print("MA2 = ", ind.MA2())
    print("MA3 = ", ind.MA3())
    print("MA4 = ", ind.MA4())
    print("MA5 = ", ind.MA5())
    print("MA6 = ", ind.MA6())
    print("MA7 = ", ind.MA7())
    print("MA8 = ", ind.MA8())
    print("MA9 = ", ind.MA9())
    print("MA10 = ", ind.MA10())
    print("MA11 = ", ind.MA11())
    print("MA12 = ", ind.MA12())
    print("MA13 = ", ind.MA13())
    print("MA14 = ", ind.MA14())
    print("MA15 = ", ind.MA15())
    print("MA16 = ", ind.MA16())
    print("MA17 = ", ind.MA17())
    print("MA18 = ", ind.MA18())
    print("MA19 = ", ind.MA19())
    print("MA20 = ", ind.MA20())
    print("MA21 = ", ind.MA21())
    print("MA22 = ", ind.MA22())
    print("MA23 = ", ind.MA23())
    print("MA24 = ", ind.MA24())
    print("MA25 = ", ind.MA25())
    print("MA26 = ", ind.MA26())
    print("MA27 = ", ind.MA27())
    print("MA28 = ", ind.MA28())
    print("MA29 = ", ind.MA29())
    print("MA30 = ", ind.MA30())
    print("MA31 = ", ind.MA31())
    print("MA32 = ", ind.MA32())
    print("MA33 = ", ind.MA33())
    print("MA34 = ", ind.MA34())
    print("MA35 = ", ind.MA35())
    print("MA36 = ", ind.MA36())
    print("MA37 = ", ind.MA37())
    print("MA38 = ", ind.MA38())
    print("MA39 = ", ind.MA39())
    print("MA40 = ", ind.MA40())
    print("MA41 = ", ind.MA41())
    print("MA42 = ", ind.MA42())
    print("MA43 = ", ind.MA43())
    print("MA44 = ", ind.MA44())
    print("MA45 = ", ind.MA45())
    print("ML1 = ", ind.ML1())
    print("ML2 = ", ind.ML2())
    print("ML3 = ", ind.ML3())
    print("ML4 = ", ind.ML4())
    print("ML5 = ", ind.ML5())
    print("ML6 = ", ind.ML6())
    print("ML7 = ", ind.ML7())
    print("ML8 = ", ind.ML8())
    print("ML9 = ", ind.ML9())
    print("ML10 = ", ind.ML10())
    print("ML11 = ", ind.ML11())
    print("ML12 = ", ind.ML12())
    print("ML13 = ", ind.ML13())
    print("ML14 = ", ind.ML14())
    print("ML15 = ", ind.ML15())
    print("ML16 = ", ind.ML16())
    print("ML17 = ", ind.ML17())
    print("ML18 = ", ind.ML18())
    print("ML19 = ", ind.ML19())
    print("ML20 = ", ind.ML20())
    print("ML21 = ", ind.ML21())
    print("ML22 = ", ind.ML22())
    print("MH1  = ", ind.MH1())
    print("MH2  = ", ind.MH2())
    print("MH3  = ", ind.MH3())
    print("MH4  = ", ind.MH4())
    print("MH5  = ", ind.MH5())
    print("MH6  = ", ind.MH6())
    print("MH7  = ", ind.MH7())
    print("MH8  = ", ind.MH8())
    print("MH9  = ", ind.MH9())
    print("MH10 = ", ind.MH10())
    print("MH11 = ", ind.MH11())
    print("MH12 = ", ind.MH12())
    print("MH13 = ", ind.MH13())
    print("MH14 = ", ind.MH14())
    print("MH15 = ", ind.MH15())
    print("MH16 = ", ind.MH16())
    print("MH17 = ", ind.MH17())
    print("MH18 = ", ind.MH18())
    print("MH19 = ", ind.MH19())
    print("MH20 = ", ind.MH20())
    print("MH21 = ", ind.MH21())
    print("MH22 = ", ind.MH22())
    print("MH23 = ", ind.MH23())
    print("MH24 = ", ind.MH24())
    print("MH25 = ", ind.MH25())
    print("MH26 = ", ind.MH26())
    print("MH27 = ", ind.MH27())
    print("FL1 = ", ind.FL1())
    print("FL2 = ", ind.FL2())
    print("FL3 = ", ind.FL3())
    print("FH1 = ", ind.FH1())
    print("FH2 = ", ind.FH2())
    print("FH3 = ", ind.FH3())
    print("FH4 = ", ind.FH4())
    print("FH5 = ", ind.FH5())
    print("FH6 = ", ind.FH6())
    print("FH7 = ", ind.FH7())
    print("FH8 = ", ind.FH8())
    print("FH9 = ", ind.FH9())
    print("FH10 = ", ind.FH10())
    print("DL1 = ", ind.DL1())
    print("DL2 = ", ind.DL2())
    print("DL3 = ", ind.DL3())
    print("DL4 = ", ind.DL4())
    print("DL5 = ", ind.DL5())
    print("DL6 = ", ind.DL6())
    print("DL7 = ", ind.DL7())
    print("DL8 = ", ind.DL8())
    print("DL9 = ", ind.DL9())
    print("DL10 = ", ind.DL10())
    print("DL11 = ", ind.DL11())
    print("DL12 = ", ind.DL12())
    print("DL13 = ", ind.DL13())
    print("DL14 = ", ind.DL14())
    print("DL15 = ", ind.DL15())
    print("DL16 = ", ind.DL16())
    print("DL17 = ", ind.DL17())
    print("DL18 = ", ind.DL18())
    print("DL19 = ", ind.DL19())
    print("DL20 = ", ind.DL20())
    print("DH1 = ", ind.DH1())
    print("DH2 = ", ind.DH2())
    print("DH3 = ", ind.DH3())
    print("DH4 = ", ind.DH4())
    print("DH5 = ", ind.DH5())
    print("DH6 = ", ind.DH6())
    print("DH7 = ", ind.DH7())
    print("DH8 = ", ind.DH8())
    print("DH9 = ", ind.DH9())
    print("DH10 = ", ind.DH10())
    print("DH11 = ", ind.DH11())
    print("DH12 = ", ind.DH12())
    print("DH13 = ", ind.DH13())
    print("DH14 = ", ind.DH14())
    print("DH15 = ", ind.DH15())
    print("DH16 = ", ind.DH16())
    print("DH17 = ", ind.DH17())
    print("DH18 = ", ind.DH18())
    print("DH19 = ", ind.DH19())
    print("DH20 = ", ind.DH20())
    print("DH21 = ", ind.DH21())
    print("TA1 = ", ind.TA1())
    print("TA2 = ", ind.TA2())
    print("TL1 = ", ind.TL1())
    print("TL2 = ", ind.TL2())
    print("TH1 = ", ind.TH1())
    print("TH2 = ", ind.TH2())
    print("RA1 = ", ind.RA1())
    print("RA2 = ", ind.RA2())
    print("RA3 = ", ind.RA3())
    print("RA4 = ", ind.RA4())
    print("RA5 = ", ind.RA5())
    print("RA6 = ", ind.RA6())
    print("RA7 = ", ind.RA7())
    print("RA8 = ", ind.RA8())
    print("RA9 = ", ind.RA9())
