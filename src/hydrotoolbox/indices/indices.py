# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tstoolbox import tsutils


class Indices:
    def __init__(self, data, use_median=False, water_year="A-SEP"):
        if isinstance(data, pd.DataFrame):
            if len(data.columns) != 1:
                raise ValueError(
                    tsutils.error_wrapper(
                        f"""
Can only calculate indices on 1 series, you gave {len(data.columns)}.
                                                       """
                    )
                )

        self.use_median = use_median
        self.water_year = water_year

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
        return self.data.mean()

    def MA2(self):
        return self.data.median()

    def MA3(self):
        tmpdata = (
            self.data.groupby(pd.Grouper(freq=self.water_year)).std().mean()
            / self.MA1()
        )
        if self.use_median is True:
            return tmpdata.median() * 100
        return tmpdata.mean() * 100

    def MA4(self):
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
        return self.data.mean() / self.data.median()

    def MA6(self):
        return self.data.quantile(0.9) / self.data.quantile(0.1)

    def MA7(self):
        return self.data.quantile(0.8) / self.data.quantile(0.2)

    def MA8(self):
        return self.data.quantile(0.75) / self.data.quantile(0.25)

    def MA9(self):
        return (self.data.quantile(0.9) - self.data.quantile(0.1)) / self.MA2()

    def MA10(self):
        return (self.data.quantile(0.8) - self.data.quantile(0.2)) / self.MA2()

    def MA11(self):
        return (self.data.quantile(0.75) - self.data.quantile(0.25)) / self.MA2()

    def _make_MA_12_23(month):
        def template(self):
            if self.use_median is True:
                return self.data[self.data.index.month == month].median()
            return self.data[self.data.index.month == month].mean()

        return template

    MA12 = _make_MA_12_23(1)
    MA13 = _make_MA_12_23(2)
    MA14 = _make_MA_12_23(3)
    MA15 = _make_MA_12_23(4)
    MA16 = _make_MA_12_23(5)
    MA17 = _make_MA_12_23(6)
    MA18 = _make_MA_12_23(7)
    MA19 = _make_MA_12_23(8)
    MA20 = _make_MA_12_23(9)
    MA21 = _make_MA_12_23(10)
    MA22 = _make_MA_12_23(11)
    MA23 = _make_MA_12_23(12)

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
    MA25 = _make_MA_24_35(2)
    MA26 = _make_MA_24_35(3)
    MA27 = _make_MA_24_35(4)
    MA28 = _make_MA_24_35(5)
    MA29 = _make_MA_24_35(6)
    MA30 = _make_MA_24_35(7)
    MA31 = _make_MA_24_35(8)
    MA32 = _make_MA_24_35(9)
    MA33 = _make_MA_24_35(10)
    MA34 = _make_MA_24_35(11)
    MA35 = _make_MA_24_35(12)

    def MA36(self):
        return (
            max(self.data_monthly_mean) - min(self.data_monthly_mean)
        ) / self.data_monthly_mean.median()

    def MA37(self):
        return (
            self.data_monthly_mean.quantile(0.75)
            - self.data_monthly_mean.quantile(0.25)
        ) / self.data_monthly_mean.median()

    def MA38(self):
        return (
            self.data_monthly_mean.quantile(0.90)
            - self.data_monthly_mean.quantile(0.10)
        ) / self.data_monthly_mean.median()

    def MA39(self):
        return self.data_monthly_mean.std() / self.data_monthly_mean.mean() * 100

    def MA40(self):
        return (
            self.data_monthly_mean.mean() - self.data_monthly_mean.median()
        ) / self.data_monthly_mean.median()

    def MA41(self, carea=1):
        return self.data_yearly_mean.mean() / carea

    def MA42(self):
        return (
            max(self.data_yearly_mean) - min(self.data_yearly_mean)
        ) / self.data_yearly_mean.median()

    def MA43(self):
        return (
            self.data_yearly_mean.quantile(0.75) - self.data_yearly_mean.quantile(0.25)
        ) / self.data_yearly_mean.median()

    def MA44(self):
        return (
            self.data_yearly_mean.quantile(0.90) - self.data_yearly_mean.quantile(0.10)
        ) / self.data_yearly_mean.median()

    def MA45(self):
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
    ML2 = _make_ML_1_12(2)
    ML3 = _make_ML_1_12(3)
    ML4 = _make_ML_1_12(4)
    ML5 = _make_ML_1_12(5)
    ML6 = _make_ML_1_12(6)
    ML7 = _make_ML_1_12(7)
    ML8 = _make_ML_1_12(8)
    ML9 = _make_ML_1_12(9)
    ML10 = _make_ML_1_12(10)
    ML11 = _make_ML_1_12(11)
    ML12 = _make_ML_1_12(12)

    def ML13(self):
        return self.data_monthly_min.std() / self.data_monthly_min.mean() * 100

    def ML14(self):
        return (
            self.data_yearly_min
            / self.data.groupby(pd.Grouper(freq=self.water_year)).median()
        ).mean()

    def ML15(self):
        return (self.data_yearly_min / self.data_yearly_mean).mean()

    def ML16(self):
        return (self.data_yearly_min / self.data_yearly.median()).median()

    def ML17(self):
        if self.use_median is True:
            return (
                self.data.rolling(7)
                .mean()
                .groupby(pd.Grouper(freq=self.water_year))
                .min()
                / self.data_yearly_mean
            ).median()
        return (
            self.data.rolling(7).mean().groupby(pd.Grouper(freq=self.water_year)).min()
            / self.data_yearly_mean
        ).mean()

    def ML18(self):
        ratios = (
            self.data.rolling(7).mean().groupby(pd.Grouper(freq=self.water_year))
        ).min() / self.data_yearly_mean
        return ratios.std() / ratios.mean() * 100

    def ML19(self):
        if self.use_median is True:
            return (self.data_yearly_min / self.data_yearly_mean).median() * 100
        return (self.data_yearly_min / self.data_yearly_mean).mean() * 100

    def ML20(self):
        from hydrotoolbox.hydrotoolbox import five_day

        newq = five_day(self.data)
        return (newq.sum() / self.data.sum()).iloc[0]

    def ML21(self):
        return self.data_yearly_min.std() / self.data_yearly_min.mean() * 100

    def ML22(self, carea=100):
        return self.data_yearly_min.mean() / carea

    def _make_MH_1_12(month):
        def template(self):
            if self.use_median is True:
                return self.data_monthly_max[
                    self.data_monthly_max.index.month == month
                ].median()
            return self.data_monthly_max[
                self.data_monthly_max.index.month == month
            ].mean()

        return template

    MH1 = _make_MH_1_12(1)
    MH2 = _make_MH_1_12(2)
    MH3 = _make_MH_1_12(3)
    MH4 = _make_MH_1_12(4)
    MH5 = _make_MH_1_12(5)
    MH6 = _make_MH_1_12(6)
    MH7 = _make_MH_1_12(7)
    MH8 = _make_MH_1_12(8)
    MH9 = _make_MH_1_12(9)
    MH10 = _make_MH_1_12(10)
    MH11 = _make_MH_1_12(11)
    MH12 = _make_MH_1_12(12)

    def MH13(self):
        return self.data_monthly_max.std() / self.data_monthly_max.mean() * 100

    def MH14(self):
        return (self.data_yearly_max / self.data_yearly.median()).median()

    def MH15(self):
        return self.data.quantile(0.99) / self.data.median()

    def MH16(self):
        return self.data.quantile(0.90) / self.data.median()

    def MH17(self):
        return self.data.quantile(0.75) / self.data.median()

    def MH18(self):
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

    def MH20(self, carea=1):
        if self.use_median:
            return self.data_yearly_max.median() / carea
        return self.data_yearly_max.mean() / carea

    def _make_MH_21_23(med_mult):
        def template(self):
            med = self.MA2()
            flow = self.data - med_mult * med
            flow[flow < 0] = 0.0
            if flow[flow > 0].count() == 0:
                return None

            qmean = flow.groupby(pd.Grouper(freq=self.water_year)).sum().mean()

            nevents = (flow > 0) & (flow.shift(-1) <= 0)
            nevents = nevents.groupby(pd.Grouper(freq=self.water_year)).sum().mean()

            return qmean / nevents / med

        return template

    MH21 = _make_MH_21_23(1)
    MH22 = _make_MH_21_23(3)
    MH23 = _make_MH_21_23(7)

    def _make_MH_24_26(med_mult=1, quantile=None):
        def template(self):

            if quantile is not None:
                medm = self.data.quantile(0.75)
            else:
                medm = self.data.median() * med_mult

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
                    allnp[group_name] = allnp[group_name] + 1
                    if flag == 1:
                        nnp[group_name] = nnp[group_name] + 1
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

    def FH3(self):
        thresh = 3 * self.MA2()
        _, _, pdur = self._lf(thresh, ">")
        if self.use_median is True:
            return pdur.median()
        return pdur.mean()

    def FH4(self):
        thresh = 7 * self.MA2()
        _, _, pdur = self._lf(thresh, ">")
        if self.use_median is True:
            return pdur.median()
        return pdur.mean()

    def FH5(self):
        thresh = self.MA2()
        nnp, _, _ = self._lf(thresh, ">")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def FH6(self):
        thresh = self.MA2() * 3
        nnp, _, _ = self._lf(thresh, ">")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def FH7(self):
        thresh = self.MA2() * 7
        nnp, _, _ = self._lf(thresh, ">")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def FH8(self):
        thresh = self.data.quantile(0.75)
        nnp, _, _ = self._lf(thresh, ">")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def FH9(self):
        thresh = self.data.quantile(0.25)
        nnp, _, _ = self._lf(thresh, ">")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def FH10(self):
        thresh = self.data_yearly.min().median()
        nnp, _, _ = self._lf(thresh, ">")
        if self.use_median is True:
            return nnp.median()
        return nnp.mean()

    def FH11(self):
        return 1.0

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

    def DL2(self):
        return self._roll(3, "min")

    def DL3(self):
        return self._roll(7, "min")

    def DL4(self):
        return self._roll(30, "min")

    def DL5(self):
        return self._roll(90, "min")

    def DL6(self):
        stat = self._preroll(1, "min")
        return stat.std() / stat.mean() * 100

    def DL7(self):
        stat = self._preroll(3, "min")
        return stat.std() / stat.mean() * 100

    def DL8(self):
        stat = self._preroll(7, "min")
        return stat.std() / stat.mean() * 100

    def DL9(self):
        stat = self._preroll(30, "min")
        return stat.std() / stat.mean() * 100

    def DL10(self):
        stat = self._preroll(90, "min")
        return stat.std() / stat.mean() * 100

    def DL11(self):
        return self.data_yearly_min.mean() / self.data.median()

    def DL12(self):
        stat = self._preroll(7, "min")
        return stat.mean() / self.data.median()

    def DL13(self):
        stat = self._preroll(30, "min")
        return stat.mean() / self.data.median()

    def DL14(self):
        return self.data.quantile(0.25) / self.data.median()

    def DL15(self):
        return self.data.quantile(0.1) / self.data.median()

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
            if self.use_median is True:
                return stat.median()
            return stat.mean()
        return 0

    def DL19(self):
        stat = (
            self.data[self.data == 0].groupby(pd.Grouper(freq=self.water_year)).count()
        )
        return stat.std() / stat.mean() * 100

    def DL20(self):
        stat = self.data_monthly_mean[self.data_monthly_mean == 0].count()
        return stat

    def DH1(self):
        stat = self.data_yearly.max()
        if self.use_median:
            stat = stat.median()
        else:
            stat = stat.mean()
        return stat

    def DH2(self):
        return self._roll(3, "max")

    def DH3(self):
        return self._roll(7, "max")

    def DH4(self):
        return self._roll(30, "max")

    def DH5(self):
        return self._roll(90, "max")

    def DH6(self):
        stat = self._preroll(1, "max")
        return stat.std() / stat.mean() * 100

    def DH7(self):
        stat = self._preroll(3, "max")
        return stat.std() / stat.mean() * 100

    def DH8(self):
        stat = self._preroll(7, "max")
        return stat.std() / stat.mean() * 100

    def DH9(self):
        stat = self._preroll(30, "max")
        return stat.std() / stat.mean() * 100

    def DH10(self):
        stat = self._preroll(90, "max")
        return stat.std() / stat.mean() * 100

    def DH11(self):
        return self.data_yearly_max.mean() / self.data.median()

    def DH12(self):
        stat = self._preroll(7, "max")
        return stat.mean() / self.data.median()

    def DH13(self):
        stat = self._preroll(30, "max")
        return stat.mean() / self.data.median()

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

    def DH22(self):
        return None

    def DH23(self):
        return None

    def DH24(self):
        return None

    def _pre_ta1_ta2(self):
        nrows = 11
        lq = self.data.apply(np.log10)
        lma1 = np.log10(self.MA1())

        lq[self.data == 0.0] = np.log10(0.01)

        table = []
        table.append(lq[lq < 0.1 * lma1])
        table.append(lq[(lq >= 0.1 * lma1) & (lq < 0.25 * lma1)])
        table.append(lq[(lq >= 0.25 * lma1) & (lq < 0.5 * lma1)])
        table.append(lq[(lq >= 0.5 * lma1) & (lq < 0.75 * lma1)])
        table.append(lq[(lq >= 0.75 * lma1) & (lq < lma1)])
        table.append(lq[(lq >= lma1) & (lq < 1.25 * lma1)])
        table.append(lq[(lq >= 1.25 * lma1) & (lq < 1.5 * lma1)])
        table.append(lq[(lq >= 1.5 * lma1) & (lq < 1.75 * lma1)])
        table.append(lq[(lq >= 1.75 * lma1) & (lq < 2.0 * lma1)])
        table.append(lq[(lq >= 2.0 * lma1) & (lq < 2.25 * lma1)])
        table.append(lq[(lq >= 2.25 * lma1)])

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
            ldf = []
            for day in range(1, 366):
                ldf.append(df[df.index.dayofyear == day].count())
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

    def TA3(self):
        return None

    def TL1(self):
        return None

    def TL2(self):
        return None

    def TL3(self):
        return None

    def TL4(self):
        return None

    def TH1(self):
        return None

    def TH2(self):
        return None

    def TH3(self):
        return None

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
    print("FH11 = ", ind.FH11())
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
    print("DH22 = ", ind.DH22())
    print("DH23 = ", ind.DH23())
    print("DH24 = ", ind.DH24())
    print("TA1 = ", ind.TA1())
    print("TA2 = ", ind.TA2())
    print("TA3 = ", ind.TA3())
    print("TL1 = ", ind.TL1())
    print("TL2 = ", ind.TL2())
    print("TL3 = ", ind.TL3())
    print("TL4 = ", ind.TL4())
    print("TH1 = ", ind.TH1())
    print("TH2 = ", ind.TH2())
    print("TH3 = ", ind.TH3())
    print("RA1 = ", ind.RA1())
    print("RA2 = ", ind.RA2())
    print("RA3 = ", ind.RA3())
    print("RA4 = ", ind.RA4())
    print("RA5 = ", ind.RA5())
    print("RA6 = ", ind.RA6())
    print("RA7 = ", ind.RA7())
    print("RA8 = ", ind.RA8())
    print("RA9 = ", ind.RA9())
