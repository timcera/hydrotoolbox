import numpy as np


def load_streamflow(path):
    """load streamflow into memory

    Args:
        path (str|DataFrame): path of streamflow csv file, or pandas DataFrame

    Returns:
        tuple: (date of np.datetime64, streamflow of float)
    """
    if isinstance(path, str):
        date, Q = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            unpack=True,
            dtype=[("date", "datetime64[D]"), ("Q", float)],
            converters={0: np.datetime64},
            encoding="utf8",
        )
        year = date.astype("datetime64[Y]").astype(int) + int(
            str(np.datetime64(0, "Y"))
        )
        month = date.astype("datetime64[M]").astype(int) % 12 + 1
        day = (date - date.astype("datetime64[M]")).astype(int) + 1
        date = np.rec.fromarrays(
            [year, month, day], dtype=[("Y", "i4"), ("M", "i4"), ("D", "i4")]
        )
    else:
        df_date = path.iloc[:, 0].astype("datetime64")
        date = np.rec.fromarrays(
            [df_date.dt.year, df_date.dt.month, df_date.dt.day],
            dtype=[("Y", "i4"), ("M", "i4"), ("D", "i4")],
        )
        Q = path.iloc[:, 1].values.astype(float)
    return clean_streamflow(date, Q)


def clean_streamflow(date, Q):
    Q[np.isnan(Q)] = 0
    Q = np.abs(Q)
    year = date["Y"]
    year_unique = np.unique(year)
    year_delete = clean_streamflow_jit(year, year_unique, Q)
    idx_delete = np.isin(year, year_delete)
    return Q[~idx_delete], date[~idx_delete]


def clean_streamflow_jit(year, year_unique, Q):
    return [y for y in year_unique if (Q[year == y] >= 0).sum() < 120]


def moving_average(x, w):
    res = np.convolve(x, np.ones(w)) / w
    return res[w - 1 : -w + 1]


def multi_arange_steps(starts, stops, steps):
    pos = 0
    cnt = np.sum((stops - starts + steps - np.sign(steps)) // steps, dtype=np.int64)
    res = np.zeros((cnt,), dtype=np.int64)
    for i in range(starts.size):
        v, stop, step = starts[i], stops[i], steps[i]
        if step > 0:
            while v < stop:
                res[pos] = v
                pos += 1
                v += step
        elif step < 0:
            while v > stop:
                res[pos] = v
                pos += 1
                v += step
    assert pos == cnt
    return res


def multi_arange(starts, stops):
    pos = 0
    cnt = np.sum(stops - starts, dtype=np.int64)
    res = np.zeros((cnt,), dtype=np.int64)
    for i in range(starts.size):
        num = stops[i] - starts[i]
        res[pos : pos + num] = np.arange(starts[i], stops[i])
        pos += num
    return res


def NSE(q_obs, q_sim):
    ss_res = np.sum(np.square(q_obs - q_sim))
    ss_tot = np.sum(np.square(q_obs - np.mean(q_obs)))
    return (1 - ss_res / (ss_tot + 1e-10)) - 1e-10
