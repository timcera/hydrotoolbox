"""Tools for hydrology.

hydrotoolbox baseflow_sep --area 59.1 --area_units 'mile**2' linear < daily.csv
hydrotoolbox baseflow_sep sliding < daily.csv
hydrotoolbox baseflow_sep eckardt,sliding < daily.csv
...

hydrotoolbox recession """

import pandas as pd


def five_day(Q):
    # Create new DataFrame with arbitrary daily index
    ndf = pd.DataFrame(
        Q, index=pd.date_range(start="1/1/2000", periods=len(Q), freq="D")
    )
    vals = ndf.groupby(pd.Grouper(freq="5D")).min().astype("float64")
    srccol = 0.9 * vals
    prevrow = vals.shift(-1)
    nextrow = vals.shift(1)
    mask = (srccol > prevrow) | (srccol > nextrow)

    vals[mask] = None
    vals = vals.interpolate(method="linear")

    vals = vals.reindex(ndf.index).ffill()

    mask = vals[0].values > Q
    vals.loc[mask, 0] = Q[mask]

    return vals[0].values
