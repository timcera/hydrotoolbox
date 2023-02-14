import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from toolbox_utils import tsutils


class Indices:
    def __init__(self, data, use_median=False, water_year="A-SEP", drainage_area=1):
        if isinstance(data, pd.DataFrame) and len(data.columns) != 1:
            raise ValueError(
                tsutils.error_wrapper(
                    f"""
                    Can only calculate indices on 1 series, you gave
                    {len(data.columns)}.
                    """
                )
            )

        self.use_median = use_median
        self.water_year = water_year
        self.drainage_area = float(drainage_area)

        self.data = pd.Series(data.iloc[:, 0].values, index=data.index)
        self.data[self.data < 0] = pd.NA
        self.data = self.data.dropna()

        self.data_monthly = self.data.groupby(pd.Grouper(freq="M"))
        self.data_yearly = self.data.groupby(pd.Grouper(freq=self.water_year))

        self.data_monthly_mean = self.data_monthly.mean()
        self.data_yearly_mean = self.data_yearly.mean()

        self.data_monthly_min = self.data_monthly.min()
        self.data_yearly_min = self.data_yearly.min()

        self.data_monthly_max = self.data_monthly.max()
        self.data_yearly_max = self.data_yearly.max()

        self.log10data = np.log10(self.data)

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
        tmpdata = (
            self.data.groupby(pd.Grouper(freq=self.water_year)).std().mean()
            / self.MA1()
        )
        if self.use_median is True:
            return tmpdata.median() * 100
        return tmpdata.mean() * 100

    def MA4(self):
        """MA4
        Standard deviation of the percentiles of the logs of the entire flow
        record divided by the mean of percentiles of the logs.   Compute the
        log10 of the daily flows for the entire record.  Compute the 5th, 10th,
        15th, 20th, 25th, 30th, 35th, 40th, 45th, 50th, 55th, 60th, 65th, 70th,
        75th, 80th, 85th, 90th, and 95th  percentiles for the logs of the
        entire flow record.

        Percentiles are computed by interpolating between the ordered
        (ascending) logs of the flow values. Compute the standard    deviation
        and mean for the percentile values. Divide the standard deviation by
        the mean.
        percent–spatial"""
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

    def _make_MA_6_8(high, low):
        def template(self):
            return self.data.quantile(high) / self.data.quantile(low)

        return template

    MA6 = _make_MA_6_8(0.9, 0.1)
    MA6.__doc__ = """MA6
Range in daily flows is the ratio of the 10-percent to 90-percent  exceedance
values for the entire flow record. Compute the 5-percent to 95-percent
exceedance values for the entire flow record. Exceedance is computed by
interpolating between the   ordered (descending) flow values.  Divide the
10-percent exceedance value by the 90-percent value.
dimensionless—spatial"""
    MA7 = _make_MA_6_8(0.8, 0.2)
    MA7.__doc__ = """MA7
Range in daily flows is computed like MA6, except using the 20 percent and 80
percent exceedance values. Divide the 20 percent exceedance value by the 80
percent value.
dimensionless—spatial"""
    MA8 = _make_MA_6_8(0.75, 0.25)
    MA8.__doc__ = """MA8
Range in daily flows is computed like MA6, except using the 25-percent and
75-percent exceedance values. Divide the 25-percent exceedance value by the
75-percent value.
dimensionless—spatial"""

    def _make_MA_9_11(high, low):
        def template(self):
            return (self.data.quantile(high) - self.data.quantile(low)) / self.MA2()

        return template

    MA9 = _make_MA_9_11(0.9, 0.1)
    MA9.__doc__ = """MA9
Spread in daily flows is the ratio of the difference between the 90th and 10th
percentile of the logs of the flow data to the log of the median of the entire
flow record. Compute the log10 of the daily flows for the entire record.
Compute the 5th, 10th, 15th, 20th, 25th, 30th, 35th, 40th, 45th, 50th, 55th,
60th, 65th, 70th, 75th, 80th, 85th, 90th, and 95th percentiles for the logs of
the entire flow record. Percentiles are computed by interpolating between the
ordered (ascending) logs of the flow values.  Compute MA9 as (90th –10th)
/log10(MA2).
dimensionless—spatial"""
    MA10 = _make_MA_9_11(0.8, 0.2)
    MA10.__doc__ = """MA10
Spread in daily flows is computed like MA9, except using the 20th and 80th
percentiles.
dimensionless—spatial"""
    MA11 = _make_MA_9_11(0.75, 0.25)
    MA11.__doc__ = """MA11
Spread in daily flows is computed like MA9, except using the 25th and 75th
percentiles.
dimensionless—spatial"""

    def _make_MA_12_23(month):
        def template(self):
            if self.use_median is True:
                return self.data[self.data.index.month == month].median()
            return self.data[self.data.index.month == month].mean()

        return template

    MA12 = _make_MA_12_23(1)
    MA12.__doc__ = """MA12
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA12 is the mean of all January flow values over the entire record
(cubic feet per second— temporal)."""
    MA13 = _make_MA_12_23(2)
    MA13.__doc__ = """MA13
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA13 is the mean of all February flow values over the entire record
MA12 is the mean of all January flow values over the entire record
(cubic feet per second— temporal)."""
    MA14 = _make_MA_12_23(3)
    MA14.__doc__ = """MA14
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA14 is the mean of all March flow values over the entire record
MA12 is the mean of all January flow values over the entire record
(cubic feet per second— temporal)."""
    MA15 = _make_MA_12_23(4)
    MA15.__doc__ = """MA15
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA15 is the mean of all April flow values over the entire record
MA12 is the mean of all January flow values over the entire record
(cubic feet per second— temporal)."""
    MA16 = _make_MA_12_23(5)
    MA16.__doc__ = """MA16
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA16 is the mean of all May flow values over the entire record
MA12 is the mean of all January flow values over the entire record
(cubic feet per second— temporal)."""
    MA17 = _make_MA_12_23(6)
    MA17.__doc__ = """MA17
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA17 is the mean of all June flow values over the entire record
(cubic feet per second— temporal)."""
    MA18 = _make_MA_12_23(7)
    MA18.__doc__ = """MA18
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA18 is the mean of all July flow values over the entire record
(cubic feet per second— temporal)."""
    MA19 = _make_MA_12_23(8)
    MA19.__doc__ = """MA19
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA19 is the mean of all August flow values over the entire record
(cubic feet per second— temporal)."""
    MA20 = _make_MA_12_23(9)
    MA20.__doc__ = """MA20
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA20 is the mean of all September flow values over the entire record
(cubic feet per second— temporal)."""
    MA21 = _make_MA_12_23(10)
    MA21.__doc__ = """MA21
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA21 is the mean of all October flow values over the entire record
(cubic feet per second— temporal)."""
    MA22 = _make_MA_12_23(11)
    MA22.__doc__ = """MA22
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA22 is the mean of all November flow values over the entire record
(cubic feet per second— temporal)."""
    MA23 = _make_MA_12_23(12)
    MA23.__doc__ = """MA23
Means (or medians) of monthly flow values. Compute the means for each.  Means
(or medians) of monthly flow values. Compute the means for each month over the
entire flow record.

MA23 is the mean of all December flow values over the entire record
(cubic feet per second— temporal)."""

    def _make_MA_24_35(month):
        def template(self):
            tmpdata = (
                self.data.groupby(pd.Grouper(freq="M")).std()
                / self.data.groupby(pd.Grouper(freq="M")).mean()
            )
            if self.use_median is True:
                return tmpdata[tmpdata.index.month == month].median() * 100
            return tmpdata[tmpdata.index.month == month].mean() * 100

        return template

    MA24 = _make_MA_24_35(1)
    MA24.__doc__ = """MA24
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA24 is the variability of all January flow values over the entire record.
percent—temporal"""
    MA25 = _make_MA_24_35(2)
    MA25.__doc__ = """MA25
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA25 is the variability of all February flow values over the entire record.
percent—temporal"""
    MA26 = _make_MA_24_35(3)
    MA26.__doc__ = """MA26
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA26 is the variability of all March flow values over the entire record.
percent—temporal"""
    MA27 = _make_MA_24_35(4)
    MA27.__doc__ = """MA27
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA27 is the variability of all April flow values over the entire record.
percent—temporal"""
    MA28 = _make_MA_24_35(5)
    MA28.__doc__ = """MA28
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA28 is the variability of all May flow values over the entire record.
percent—temporal"""
    MA29 = _make_MA_24_35(6)
    MA29.__doc__ = """MA29
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA29 is the variability of all June flow values over the entire record.
percent—temporal"""
    MA30 = _make_MA_24_35(7)
    MA30.__doc__ = """MA30
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA30 is the variability of all July flow values over the entire record.
percent—temporal"""
    MA31 = _make_MA_24_35(8)
    MA31.__doc__ = """MA31
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA31 is the variability of all August flow values over the entire record.
percent—temporal"""
    MA32 = _make_MA_24_35(9)
    MA32.__doc__ = """MA32
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA32 is the variability of all September flow values over the entire record.
percent—temporal"""
    MA33 = _make_MA_24_35(10)
    MA33.__doc__ = """
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA33 is the variability of all October flow values over the entire record.
percent—temporal"""
    MA34 = _make_MA_24_35(11)
    MA34.__doc__ = """MA34
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA34 is the variability of all November flow values over the entire record.
percent—temporal"""
    MA35 = _make_MA_24_35(12)
    MA35.__doc__ = """MA35
Variability (coefficient of variation) of monthly flow values. Compute the
standard deviation for each.  Variability (coefficient of month in each year
over the entire flow record. Divide the standard deviation by the mean for each
month. Average (or take median of) these values for each month across all
years.

MA35 is the variability of all December flow values over the entire record.
percent—temporal"""

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

    def _make_MA_37_38(high, low):
        def template(self):
            return (
                self.data_monthly_mean.quantile(high)
                - self.data_monthly_mean.quantile(low)
            ) / self.data_monthly_mean.median()

        return template

    MA37 = _make_MA_37_38(0.75, 0.25)
    MA37.__doc__ = """MA37
Variability across monthly flows. Compute the first (25th percentile) and the
third (75th percentile) quartiles (every month in dimensionless— the flow
record). MA37 is the third quartile minus the first quartile divided by the
median of the monthly means.
dimensionless-spatial"""
    MA38 = _make_MA_37_38(0.9, 0.1)
    MA38.__doc__ = """MA38
Variability across monthly flows. Compute the 10th and 90th percentiles for the
monthly means (every month in the flow record). MA38 is the 90th percentile
minus the 10th percentile divided by the median of the monthly means.
dimensionless—spatial"""

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

    def _make_MA_43_44(high, low):
        def template(self):
            return (
                self.data_yearly_mean.quantile(high)
                - self.data_yearly_mean.quantile(low)
            ) / self.data_yearly_mean.median()

        return template

    MA43 = _make_MA_43_44(0.75, 0.25)
    MA43.__doc__ = """MA43
Variability across annual flows. Compute the first (25th percentile) and third
(75th percentile) quartiles for the annual means (every year in the flow
record).

MA43 is the third quartile minus the first quartile divided by the median of
the annual means.
dimensionless-spatial"""
    MA44 = _make_MA_43_44(0.9, 0.1)
    MA44.__doc__ = """MA44
Variability across annual flows. Compute the 10th and 90th percentiles for the
annual means (every year in the flow record).

MA44 is the 90th percentile minus the 10th percentile divided by the median of
the annual means.
dimensionless-spatial"""

    def MA45(self):
        """MA45
        Skewness in the annual flows. MA45 is the mean of the annual flow means
        minus the median of the annual means divided by the median of the
        annual means.
        dimensionless-spatial"""
        return (
            self.data_yearly_mean.mean() - self.data_yearly_mean.median()
        ) / self.data_yearly_mean.median()

    def _make_ML_1_12(month):
        def template(self):
            if self.use_median is True:
                return self.data_monthly_min[
                    self.data_monthly_min.index.month == month
                ].median()
            return self.data_monthly_min[
                self.data_monthly_min.index.month == month
            ].mean()

        return template

    ML1 = _make_ML_1_12(1)
    ML1.__doc__ = """ML1
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML1 is the mean of the minimums of all January flow values over the entire
record.
cubic feet per second—temporal"""
    ML2 = _make_ML_1_12(2)
    ML2.__doc__ = """ML2
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML2 is the mean of the minimums of all February flow values over the entire
record.
cubic feet per second—temporal"""
    ML3 = _make_ML_1_12(3)
    ML3.__doc__ = """ML3
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML3 is the mean of the minimums of all March flow values over the entire
record.
cubic feet per second—temporal"""
    ML4 = _make_ML_1_12(4)
    ML4.__doc__ = """ML4
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML4 is the mean of the minimums of all April flow values over the entire
record.
cubic feet per second—temporal"""
    ML5 = _make_ML_1_12(5)
    ML5.__doc__ = """ML5
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML5 is the mean of the minimums of all May flow values over the entire
record.
cubic feet per second—temporal"""
    ML6 = _make_ML_1_12(6)
    ML6.__doc__ = """ML6
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML6 is the mean of the minimums of all June flow values over the entire
record.
cubic feet per second—temporal"""
    ML7 = _make_ML_1_12(7)
    ML7.__doc__ = """ML7
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML7 is the mean of the minimums of all July flow values over the entire
record.
cubic feet per second—temporal"""
    ML8 = _make_ML_1_12(8)
    ML8.__doc__ = """ML8
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML8 is the mean of the minimums of all August flow values over the entire
record.
cubic feet per second—temporal"""
    ML9 = _make_ML_1_12(9)
    ML9.__doc__ = """ML9
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML9 is the mean of the minimums of all September flow values over the entire
record.
cubic feet per second—temporal"""
    ML10 = _make_ML_1_12(10)
    ML10.__doc__ = """ML10
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML10 is the mean of the minimums of all October flow values over the entire
record.
cubic feet per second—temporal"""
    ML11 = _make_ML_1_12(11)
    ML11.__doc__ = """ML11
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML11 is the mean of the minimums of all November flow values over the entire
record.
cubic feet per second—temporal"""
    ML12 = _make_ML_1_12(12)
    ML12.__doc__ = """ML12
Mean (or median) of minimum flows for each month across all years. Compute the
minimums for each month over the entire flow record.

ML12 is the mean of the minimums of all December flow values over the entire
record.
cubic feet per second—temporal"""

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
        if self.use_median is True:
            return stat.median()
        return stat.mean()

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

    def _make_MH_1_12(month):
        def template(self):
            stat = self.data_monthly_max[self.data_monthly_max.index.month == month]
            if self.use_median is True:
                return stat.median()
            return stat.mean()

        return template

    MH1 = _make_MH_1_12(1)
    MH1.__doc__ = """MH1
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH1 is the mean of the maximums of all January flow values over the entire
record.
second—temporal"""
    MH2 = _make_MH_1_12(2)
    MH2.__doc__ = """MH2
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH2 is the mean of the maximums of all February flow values over the entire
record.
second—temporal"""
    MH3 = _make_MH_1_12(3)
    MH3.__doc__ = """MH3
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH3 is the mean of the maximums of all March flow values over the entire
record.
second—temporal"""
    MH4 = _make_MH_1_12(4)
    MH4.__doc__ = """MH4
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH4 is the mean of the maximums of all April flow values over the entire
record.
second—temporal"""
    MH5 = _make_MH_1_12(5)
    MH5.__doc__ = """MH5
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH5 is the mean of the maximums of all May flow values over the entire
record.
second—temporal"""
    MH6 = _make_MH_1_12(6)
    MH6.__doc__ = """MH6
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH6 is the mean of the maximums of all June flow values over the entire
record.
second—temporal"""
    MH7 = _make_MH_1_12(7)
    MH7.__doc__ = """MH7
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH7 is the mean of the maximums of all July flow values over the entire
record.
second—temporal"""
    MH8 = _make_MH_1_12(8)
    MH8.__doc__ = """MH8
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH8 is the mean of the maximums of all August flow values over the entire
record.
second—temporal"""
    MH9 = _make_MH_1_12(9)
    MH9.__doc__ = """MH9
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH9 is the mean of the maximums of all September flow values over the entire
record.
second—temporal"""
    MH10 = _make_MH_1_12(10)
    MH10.__doc__ = """MH10
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH10 is the mean of the maximums of all October flow values over the entire
record.
second—temporal"""
    MH11 = _make_MH_1_12(11)
    MH11.__doc__ = """MH11
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH11 is the mean of the maximums of all November flow values over the entire
record.
second—temporal"""
    MH12 = _make_MH_1_12(12)
    MH12.__doc__ = """MH12
Mean (or median) maximum flows for each month across all years. Compute the
maximums for each month over the entire cubic feet per flow record.

MH12 is the mean of the maximums of all December flow values over the entire
record.
second—temporal"""

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

    def _MH_15_17(quant):
        def template(self):
            return self.data.quantile(quant) / self.MA2()

        return template

    MH15 = _MH_15_17(0.99)
    MH15.__doc__ = """MH15
High flow discharge index. Compute the 1-percent exceedance value for the
entire data record.

MH15 is the 1-percent exceedance value divided by the median flow for the
entire record.
dimensionless—spatial"""
    MH16 = _MH_15_17(0.90)
    MH16.__doc__ = """MH16
High flow discharge index. Compute the 1-percent exceedance value for the
entire data record.

MH16 is the 10-percent exceedance value divided by the median flow for the
entire record.
dimensionless—spatial"""
    MH17 = _MH_15_17(0.75)
    MH17.__doc__ = """MH17
High flow discharge index. Compute the 1-percent exceedance value for the
entire data record.

MH17 is the 25-percent exceedance value divided by the median flow for the
entire record.
dimensionless—spatial"""

    def MH18(self):
        """MH18
        Variability across annual maximum flows. Compute the logs (log10) of
        the maximum annual flows. Find the standard percent—spatial deviation
        and mean for these values. MH18 is the standard deviation times 100
        divided by the mean."""
        log10 = np.log10(self.data_yearly_max)
        return log10.std() / log10.mean() * 100

    def MH19(self):
        qm = np.log10(self.data_yearly_max)
        N = self.data_yearly_max.count()
        S = qm.std()
        return (
            N**2 * (qm**3).sum()
            - 3 * N * qm.sum() * (qm**2).sum()
            + 2 * (qm.sum()) ** 3
        ) / (N * (N - 1) * (N - 2) * S**3)

    def MH20(self):
        if self.use_median:
            return self.data_yearly_max.median() / self.drainage_area
        return self.data_yearly_max.mean() / self.drainage_area

    def _make_MH_21_23(med_mult):
        def template(self):
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

        return template

    MH21 = _make_MH_21_23(1)
    MH22 = _make_MH_21_23(3)
    MH23 = _make_MH_21_23(7)

    def _make_MH_24_26(med_mult=1, quantile=None):
        def template(self):
            if quantile is not None:
                medm = self.data.quantile(quantile)
            else:
                medm = self.MA2() * med_mult

            peaks, _ = find_peaks(self.data, height=medm)
            return self.data[peaks].mean() / self.MA2()

        return template

    MH24 = _make_MH_24_26(med_mult=1)
    MH25 = _make_MH_24_26(med_mult=3)
    MH26 = _make_MH_24_26(med_mult=7)
    MH27 = _make_MH_24_26(quantile=0.75)

    def _lf(self, thresh, than):
        """Returns number of events, average duration of events, count of all
        days."""
        nnp = {}
        lfdur = {}
        allnp = {}
        for group_name, group_df in self.data_yearly:
            nnp[group_name] = 0
            lfdur[group_name] = 0
            allnp[group_name] = 0
            pdur = 0
            flag = 0
            for index, value in group_df.iteritems():
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
        thresh = self.data.quantile(0.25)
        nnp, _, _ = self._lf(thresh, "<")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def FL2(self):
        thresh = self.data.quantile(0.25)
        nnp, _, _ = self._lf(thresh, "<")
        return nnp.std() / nnp.mean() * 100

    def FL3(self):
        thresh = self.data.mean() * 0.05
        nnp, _, _ = self._lf(thresh, "<")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def FH1(self):
        thresh = self.data.quantile(0.75)
        nnp, _, _ = self._lf(thresh, ">")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def FH2(self):
        thresh = self.data.quantile(0.75)
        nnp, _, _ = self._lf(thresh, ">")
        return nnp.std() / nnp.mean() * 100

    def _make_FH_3_4(med_mult):
        def template(self):
            thresh = med_mult * self.MA2()
            _, _, pdur = self._lf(thresh, ">")
            if self.use_median is True:
                return pdur.median()
            return pdur.mean()

        return template

    FH3 = _make_FH_3_4(3)
    FH4 = _make_FH_3_4(7)

    def _make_FH_5_7(med_mult):
        def template(self):
            thresh = self.MA2() * med_mult
            nnp, _, _ = self._lf(thresh, ">")
            if self.use_median is True:
                return nnp.median()
            return nnp.mean()

        return template

    FH5 = _make_FH_5_7(1)
    FH6 = _make_FH_5_7(3)
    FH7 = _make_FH_5_7(7)

    def _make_FH_8_9(quant):
        def template(self):
            thresh = self.data.quantile(quant)
            nnp, _, _ = self._lf(thresh, ">")
            if self.use_median is True:
                return nnp.median()
            return nnp.mean()

        return template

    FH8 = _make_FH_8_9(0.75)
    FH9 = _make_FH_8_9(0.25)

    def FH10(self):
        thresh = self.data_yearly.min().median()
        nnp, _, _ = self._lf(thresh, ">")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def DL1(self):
        stat = self.data_yearly.min()
        if self.use_median is True:
            return stat.median()
        return stat.mean()

    def _preroll(self, days, stattype):
        stats = []
        for group_name, vals in self.data_yearly:
            rmean = vals.rolling(days).mean()
            if stattype == "min":
                stats.append(rmean.min())
            elif stattype == "max":
                stats.append(rmean.max())
        return pd.Series(data=stats, index=range(len(stats)))

    def _roll(self, days, stattype):
        stat = self._preroll(days, stattype)
        if self.use_median is True:
            return stat.median()
        return stat.mean()

    def _make_DL_2_5_DH_2_5(days, stat):
        def template(self):
            return self._roll(days, stat)

        return template

    DL2 = _make_DL_2_5_DH_2_5(3, "min")
    DL3 = _make_DL_2_5_DH_2_5(7, "min")
    DL4 = _make_DL_2_5_DH_2_5(30, "min")
    DL5 = _make_DL_2_5_DH_2_5(90, "min")

    def _make_DL_6_10_DH_6_10(days, instat):
        def template(self):
            stat = self._preroll(days, instat)
            return stat.std() / stat.mean() * 100

        return template

    DL6 = _make_DL_6_10_DH_6_10(1, "min")
    DL7 = _make_DL_6_10_DH_6_10(3, "min")
    DL8 = _make_DL_6_10_DH_6_10(7, "min")
    DL9 = _make_DL_6_10_DH_6_10(30, "min")
    DL10 = _make_DL_6_10_DH_6_10(90, "min")

    def DL11(self):
        return self.data_yearly_min.mean() / self.MA2()

    def _make_DL_12_13_DH_12_13(days, instat):
        def template(self):
            stat = self._preroll(days, instat)
            return stat.mean() / self.MA2()

        return template

    DL12 = _make_DL_12_13_DH_12_13(7, "min")
    DL13 = _make_DL_12_13_DH_12_13(30, "min")

    def DL14(self):
        return self.data.quantile(0.25) / self.MA2()

    def DL15(self):
        return self.data.quantile(0.1) / self.MA2()

    def DL16(self):
        thresh = self.data.quantile(0.25)
        _, lfdur, _ = self._lf(thresh, "<")
        return lfdur.median()

    def DL17(self):
        thresh = self.data.quantile(0.25)
        _, lfdur, _ = self._lf(thresh, "<")
        return lfdur.std() / lfdur.mean() * 100

    def DL18(self):
        stat = (
            self.data[self.data == 0].groupby(pd.Grouper(freq=self.water_year)).count()
        )
        if any(stat):
            return stat.median() if self.use_median is True else stat.mean()
        return 0

    def DL19(self):
        stat = (
            self.data[self.data == 0].groupby(pd.Grouper(freq=self.water_year)).count()
        )
        return stat.std() / stat.mean() * 100

    def DL20(self):
        return self.data_monthly_mean[self.data_monthly_mean == 0].count()

    def DH1(self):
        stat = self.data_yearly.max()
        stat = stat.median() if self.use_median else stat.mean()
        return stat

    DH2 = _make_DL_2_5_DH_2_5(3, "max")
    DH3 = _make_DL_2_5_DH_2_5(7, "max")
    DH4 = _make_DL_2_5_DH_2_5(30, "max")
    DH5 = _make_DL_2_5_DH_2_5(90, "max")

    DH6 = _make_DL_6_10_DH_6_10(1, "max")
    DH7 = _make_DL_6_10_DH_6_10(3, "max")
    DH8 = _make_DL_6_10_DH_6_10(7, "max")
    DH9 = _make_DL_6_10_DH_6_10(30, "max")
    DH10 = _make_DL_6_10_DH_6_10(90, "max")

    def DH11(self):
        return self.data_yearly_max.mean() / self.MA2()

    DH12 = _make_DL_12_13_DH_12_13(7, "max")
    DH13 = _make_DL_12_13_DH_12_13(30, "max")

    def DH14(self):
        return self.data_monthly_mean.quantile(0.95) / self.data_monthly_mean.mean()

    def DH15(self):
        thresh = self.data.quantile(0.75)
        _, lfdur, _ = self._lf(thresh, ">")
        return lfdur.median()

    def DH16(self):
        thresh = self.data.quantile(0.75)
        _, lfdur, _ = self._lf(thresh, ">")
        return lfdur.std() / lfdur.mean() * 100

    def DH17(self):
        thresh = self.MA2()
        _, lfdur, _ = self._lf(thresh, ">")
        if self.use_median:
            return lfdur.median()
        return lfdur.mean()

    def DH18(self):
        thresh = self.MA2() * 3
        _, lfdur, _ = self._lf(thresh, ">")
        if self.use_median:
            return lfdur.median()
        return lfdur.mean()

    def DH19(self):
        thresh = self.MA2() * 7
        _, lfdur, _ = self._lf(thresh, ">")
        if self.use_median:
            return lfdur.median()
        return lfdur.mean()

    def DH20(self):
        thresh = self.data.quantile(0.75)
        _, lfdur, _ = self._lf(thresh, ">")
        if self.use_median:
            return lfdur.median()
        return lfdur.mean()

    def DH21(self):
        thresh = self.data.quantile(0.25)
        _, lfdur, _ = self._lf(thresh, ">")
        if self.use_median:
            return lfdur.median()
        return lfdur.mean()

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
        nrows, HY, HXY = self._pre_ta1_ta2()
        return 1 - (HY / np.log10(nrows))

    def TA2(self):
        nrows, HY, HXY = self._pre_ta1_ta2()
        return 100 * (1 - (HXY / np.log10(nrows)))

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
        jd[jd > 365.25] = jd[jd > 365.25] - 365.25
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
        TL2 = temp * 180 / np.pi / 360 * 365.25
        return TL2

    def TH1(self):
        """TH1
        Julian date of annual maximum. Determine the Julian date of the maximum flow
        for each year. Transform the dates to relative values on a circular scale
        (radians or degrees). Compute the x and y components for each year, and average
        them across all years. Compute the mean angle as the arc tangent of y-mean
        divided by x-mean. Transform the resultant angle back to Julian date.
        Julian day—spatial"""
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
        TH2 = temp * 180 / np.pi / 360 * 365.25
        return TH2

    def _rise_rate(self):
        delt = self.data.shift(1) - self.data
        return delt[delt < 0]

    def _fall_rate(self):
        delt = self.data.shift(1) - self.data
        return delt[delt > 0]

    def RA1(self):
        if self.use_median:
            return -self._rise_rate().median()
        return -self._rise_rate().mean()

    def RA2(self):
        rr = self._rise_rate()
        return -rr.std() / rr.mean() * 100

    def RA3(self):
        if self.use_median:
            return self._fall_rate().median()
        return self._fall_rate().mean()

    def RA4(self):
        rr = self._fall_rate()
        return rr.std() / rr.mean() * 100

    def RA5(self):
        rr = self._rise_rate()
        return len(rr) / len(self.data)

    def RA6(self):
        log = self.data.apply(np.log)
        log = log.shift(1) - log
        log = -log[log < 0]
        return log.median()

    def RA7(self):
        log = self.data.apply(np.log)
        log = log.shift(1) - log
        log = log[log > 0]
        return log.median()

    def _changes(self):
        peaks = find_peaks(self.data)[0]
        valleys = find_peaks(-self.data)[0]
        changes = np.append(peaks, valleys)
        changes = pd.DataFrame(self.data[changes])
        df = pd.DataFrame(range(len(changes.index)), index=changes.index)
        df = df.groupby(pd.Grouper(freq=self.water_year)).count()
        return df

    def RA8(self):
        df = self._changes()
        if self.use_median:
            return df.median()
        return df.mean()

    def RA9(self):
        df = self._changes()
        return df.std() / df.mean() * 100


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
