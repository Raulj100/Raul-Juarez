"""
Volume-profile balance setup (NQ futures).

Reproduces the chart layout from a Deepchart 30D / 1-tick screenshot:
  * one volume profile per session, with its POC (magenta) and value area
  * naked POCs extended to the right until price trades through them
  * multi-day balance areas (composite profiles) drawn as boxes
  * HVN / LVN zones from the composite (blue = support, red = resistance)

and adds the trade logic that the layout is built for:
  * FADE      - price probes outside a balance edge, fails to hold, and
                closes back inside -> trade back to the composite POC
  * BREAKOUT  - price closes outside the balance edge for N bars in a row
                (acceptance) -> trade to the next naked POC / prior balance

Input: 1-minute OHLCV bars with a tz-aware DatetimeIndex
       columns: open, high, low, close, volume

Run `python profile_setup.py` for a demo on synthetic data, or
`python profile_setup.py your_nq_1m.csv` with real data
(CSV columns: timestamp, open, high, low, close, volume).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
TICK = 0.25              # NQ tick size
BIN = 5.0                # profile row height in points (20 ticks)
VALUE_AREA = 0.70        # 70% value area
SESSION_TZ = "America/New_York"
SESSION_START_HOUR = 18  # CME Globex session opens 18:00 ET

BALANCE_MAX_RANGE_PCT = 0.06   # composite range must stay under 6% of price
BALANCE_MIN_SESSIONS = 3       # need at least 3 overlapping sessions
ACCEPT_BARS = 2                # 30-min closes outside edge = acceptance
FADE_MAX_BARS = 2              # probe must close back inside within N bars
MIN_PROBE = 10.0               # points a probe must exceed the edge by to count
MIN_STOP = 25.0                # points; NQ noise floor for a stop
STOP_BUFFER = 2.0              # points beyond the probe extreme


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------
@dataclass
class Profile:
    start: pd.Timestamp
    end: pd.Timestamp
    hist: pd.Series              # index = bin price (low edge), values = volume
    poc: float = field(init=False)
    vah: float = field(init=False)
    val: float = field(init=False)
    high: float = field(init=False)
    low: float = field(init=False)

    def __post_init__(self):
        self.high = float(self.hist.index.max() + BIN)
        self.low = float(self.hist.index.min())
        self.poc, self.vah, self.val = value_area(self.hist)


def build_hist(bars: pd.DataFrame) -> pd.Series:
    """Spread each bar's volume evenly across the price bins it touched."""
    lo = np.floor(bars["low"].to_numpy() / BIN) * BIN
    hi = np.floor(bars["high"].to_numpy() / BIN) * BIN
    vol = bars["volume"].to_numpy(dtype=float)
    n = ((hi - lo) / BIN).astype(int) + 1
    prices = np.concatenate([l + BIN * np.arange(k) for l, k in zip(lo, n)])
    vols = np.repeat(vol / n, n)
    return pd.Series(vols, index=prices).groupby(level=0).sum().sort_index()


def value_area(hist: pd.Series, pct: float = VALUE_AREA):
    """Classic CBOT value area: grow out from the POC two rows at a time."""
    prices, vols = hist.index.to_numpy(), hist.to_numpy()
    i_poc = int(np.argmax(vols))
    lo = hi = i_poc
    total, target = vols[i_poc], vols.sum() * pct
    while total < target and (lo > 0 or hi < len(vols) - 1):
        up = vols[hi + 1: hi + 3].sum() if hi < len(vols) - 1 else -1
        dn = vols[max(lo - 2, 0): lo].sum() if lo > 0 else -1
        if up >= dn:
            step = min(2, len(vols) - 1 - hi)
            total += vols[hi + 1: hi + 1 + step].sum(); hi += step
        else:
            step = min(2, lo)
            total += vols[lo - step: lo].sum(); lo -= step
    poc = prices[i_poc] + BIN / 2
    return float(poc), float(prices[hi] + BIN), float(prices[lo])


def session_ids(index: pd.DatetimeIndex) -> pd.Series:
    """Trading date: bars after 18:00 ET belong to the next day's session."""
    et = index.tz_convert(SESSION_TZ)
    shifted = et + pd.Timedelta(hours=24 - SESSION_START_HOUR)
    return pd.Series(shifted.date, index=index)


def session_profiles(df: pd.DataFrame) -> list[Profile]:
    out = []
    for _, bars in df.groupby(session_ids(df.index)):
        if len(bars) < 30:
            continue
        out.append(Profile(bars.index[0], bars.index[-1], build_hist(bars)))
    return out


# --------------------------------------------------------------------------
# Naked POCs, balance areas, HVN/LVN
# --------------------------------------------------------------------------
def naked_pocs(df: pd.DataFrame, profiles: list[Profile]) -> list[tuple[pd.Timestamp, float]]:
    """POCs that no later bar has traded through."""
    out = []
    for p in profiles:
        later = df.loc[df.index > p.end]
        touched = ((later["low"] <= p.poc) & (later["high"] >= p.poc)).any()
        if not touched:
            out.append((p.end, p.poc))
    return out


def balance_areas(df: pd.DataFrame, profiles: list[Profile]) -> list[Profile]:
    """
    Group consecutive sessions whose value area sits inside the running
    composite range (price keeps rotating back into it), while the
    composite range stays tight.
    Each group of >= BALANCE_MIN_SESSIONS becomes a composite profile.
    """
    groups, cur = [], [profiles[0]]
    for p in profiles[1:]:
        hi = max(x.high for x in cur + [p])
        lo = min(x.low for x in cur + [p])
        c_hi = max(x.high for x in cur)
        c_lo = min(x.low for x in cur)
        overlaps = p.val <= c_hi and p.vah >= c_lo
        tight = (hi - lo) / lo <= BALANCE_MAX_RANGE_PCT
        if overlaps and tight:
            cur.append(p)
        else:
            groups.append(cur); cur = [p]
    groups.append(cur)

    comps = []
    for g in groups:
        if len(g) >= BALANCE_MIN_SESSIONS:
            bars = df.loc[g[0].start: g[-1].end]
            comps.append(Profile(g[0].start, g[-1].end, build_hist(bars)))
    return comps


def hvn_lvn(p: Profile, smooth_pts: float = 60.0, hvn_min: float = 0.7, lvn_max: float = 0.4):
    """
    Peaks (HVN) and troughs (LVN) of the smoothed composite.
    HVN = local max above hvn_min * max volume  -> acceptance / support-resistance
    LVN = local min below lvn_max * max volume, between two HVNs -> fast-move zones
    """
    win = max(3, int(smooth_pts / BIN))
    v = p.hist.reindex(np.arange(p.hist.index.min(), p.hist.index.max() + BIN, BIN), fill_value=0)
    v = v.rolling(win, center=True, min_periods=1).mean()
    arr, px = v.to_numpy(), v.index.to_numpy() + BIN / 2
    top = arr.max()
    peaks = [i for i in range(1, len(arr) - 1)
             if arr[i] >= arr[i - 1] and arr[i] > arr[i + 1] and arr[i] >= hvn_min * top]
    merged = []                       # merge peaks closer than smooth_pts
    for i in peaks:
        if merged and px[i] - px[merged[-1]] < smooth_pts:
            if arr[i] > arr[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    peaks = merged
    hvn = [float(px[i]) for i in peaks]
    lvn = []
    for a, b in zip(peaks, peaks[1:]):
        j = a + int(np.argmin(arr[a:b + 1]))
        if arr[j] <= lvn_max * top:
            lvn.append(float(px[j]))
    return hvn, lvn


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
@dataclass
class Signal:
    time: pd.Timestamp
    kind: str          # FADE_SHORT / FADE_LONG / BREAKOUT_LONG / BREAKOUT_SHORT
    entry: float
    stop: float
    target: float

    @property
    def rr(self) -> float:
        return abs(self.target - self.entry) / max(abs(self.entry - self.stop), TICK)


def signals(df: pd.DataFrame, comp: Profile, levels: list[float], after: pd.Timestamp,
            until: pd.Timestamp | None = None) -> list[Signal]:
    """
    Scan 30-min bars between `after` and `until` for fade / breakout triggers.
    `levels` = reference prices outside this balance (naked POCs, other
    balances' POC/edges) used as breakout targets.
    """
    window = df.loc[df.index > after] if until is None else df.loc[(df.index > after) & (df.index < until)]
    b = window.resample("30min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    hi, lo, poc = comp.high, comp.low, comp.poc
    out: list[Signal] = []

    def long_stop(entry, extreme):
        return min(extreme - STOP_BUFFER, entry - MIN_STOP)

    def short_stop(entry, extreme):
        return max(extreme + STOP_BUFFER, entry + MIN_STOP)

    def next_level(entry, risk, up):
        cands = [x for x in levels if (x >= entry + risk if up else x <= entry - risk)]
        if cands:
            return min(cands) if up else max(cands)
        return entry + (hi - lo) / 2 if up else entry - (hi - lo) / 2  # measured move

    above = below = 0                 # consecutive closes outside each edge
    probe_hi = probe_lo = None        # extreme of the current excursion
    for t, r in b.iterrows():
        # track the excursion outside the balance
        if r.high > hi + MIN_PROBE:
            probe_hi = r.high if probe_hi is None else max(probe_hi, r.high)
        if r.low < lo - MIN_PROBE:
            probe_lo = r.low if probe_lo is None else min(probe_lo, r.low)
        above = above + 1 if r.close > hi else 0
        below = below + 1 if r.close < lo else 0

        # acceptance -> breakout (stop goes back inside the balance)
        if above == ACCEPT_BARS:
            stop = long_stop(r.close, hi - MIN_STOP)
            out.append(Signal(t, "BREAKOUT_LONG", r.close, stop, next_level(r.close, r.close - stop, True)))
        if below == ACCEPT_BARS:
            stop = short_stop(r.close, lo + MIN_STOP)
            out.append(Signal(t, "BREAKOUT_SHORT", r.close, stop, next_level(r.close, stop - r.close, False)))

        # rejection -> fade back to composite POC
        # (probe outside, then a close back inside before acceptance builds)
        if probe_hi is not None and r.close <= hi:
            if above == 0 and 0 < _bars_outside(b, t, hi, True) <= FADE_MAX_BARS + 1:
                out.append(Signal(t, "FADE_SHORT", r.close, short_stop(r.close, probe_hi), poc))
            probe_hi = None
        if probe_lo is not None and r.close >= lo:
            if below == 0 and 0 < _bars_outside(b, t, lo, False) <= FADE_MAX_BARS + 1:
                out.append(Signal(t, "FADE_LONG", r.close, long_stop(r.close, probe_lo), poc))
            probe_lo = None
    return out


def _bars_outside(b: pd.DataFrame, t, edge: float, up: bool) -> int:
    """How many bars up to and including t poked through `edge` in this excursion."""
    n = 0
    for _, r in b.loc[:t].iloc[::-1].iterrows():
        if (r.high > edge) if up else (r.low < edge):
            n += 1
        else:
            break
    return n


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def plot(df, profiles, comps, npocs, sigs, path="profile_setup.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(18, 8), facecolor="#14161d")
    ax.set_facecolor("#14161d")
    x_of = {p.start: i for i, p in enumerate(profiles)}
    width = 0.9

    def xpos(ts):
        return max(i for p, i in x_of.items() if p <= ts) if ts >= profiles[0].start else 0

    # composite balance boxes + HVN/LVN zones
    for c in comps:
        x0, x1 = xpos(c.start), xpos(c.end) + 1
        ax.add_patch(Rectangle((x0, c.low), x1 - x0, c.high - c.low,
                               fill=False, ec="#cfd3dc", lw=1, ls="--"))
        ax.hlines([c.vah, c.val], x0, len(profiles), colors="#cfd3dc", lw=0.6, ls=":")
        hvn, lvn = hvn_lvn(c)
        for h in hvn:
            ax.axhspan(h - BIN * 2, h + BIN * 2, color="#3b4a6b", alpha=0.35)
        for l in lvn:
            ax.axhspan(l - BIN * 2, l + BIN * 2, color="#6b2f3b", alpha=0.35)
        ch = c.hist / c.hist.max() * (x1 - x0) * 0.5
        ax.barh(ch.index + BIN / 2, ch.values, height=BIN, left=x0,
                color="#8a93ad", alpha=0.18)

    # session profiles
    for i, p in enumerate(profiles):
        h = p.hist / p.hist.max() * width
        in_va = (h.index >= p.val) & (h.index < p.vah)
        ax.barh(h.index + BIN / 2, h.values, height=BIN, left=i,
                color=np.where(in_va, "#d9dbe3", "#6c7a9c"))
        ax.vlines(i, p.low, p.high, colors="white", lw=0.8)
        ax.vlines(i + width, p.val, p.vah, colors="#e3c728", lw=2.5)
        ax.hlines(p.poc, i, i + width * 0.8, colors="#ff2bd6", lw=2)

    # naked POCs
    for ts, px in npocs:
        ax.hlines(px, xpos(ts) + 1, len(profiles), colors="#c21fe0", lw=1)

    # signals
    marks = {"FADE_SHORT": ("v", "#ff4d4d"), "FADE_LONG": ("^", "#3ddc84"),
             "BREAKOUT_LONG": ("^", "#2fa8ff"), "BREAKOUT_SHORT": ("v", "#ff9f1a")}
    for s in sigs:
        m, col = marks[s.kind]
        ax.scatter(xpos(s.time) + 0.5, s.entry, marker=m, s=120, color=col, zorder=5)

    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels([p.end.tz_convert(SESSION_TZ).strftime("%d/%m") for p in profiles],
                       color="#9aa0ad", fontsize=8)
    ax.yaxis.tick_right(); ax.tick_params(colors="#9aa0ad")
    ax.set_xlim(-0.5, len(profiles) + 0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=110)
    return path


# --------------------------------------------------------------------------
# Demo data / entry point
# --------------------------------------------------------------------------
def synthetic_nq(days: int = 15, seed: int = 7) -> pd.DataFrame:
    """Trend -> balance -> rally-to-edge path, roughly like the screenshot."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-07-13 22:00", periods=days * 1440, freq="1min", tz="UTC")
    anchors = np.interp(np.linspace(0, 1, len(idx)),
                        [0, .12, .25, .45, .52, .62, .75, .85, .90, .935, .965, 1],
                        [29700, 28700, 29300, 28500, 27350, 28700, 27500, 28000, 27350, 28450, 28880, 28600])
    noise = np.cumsum(rng.normal(0, 3, len(idx)))
    noise -= pd.Series(noise).rolling(240, min_periods=1).mean().to_numpy()
    close = np.round((anchors + noise) / TICK) * TICK
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 4, len(idx)))
    df = pd.DataFrame({
        "open": open_, "close": close,
        "high": np.maximum(open_, close) + np.round(wick / TICK) * TICK,
        "low": np.minimum(open_, close) - np.round(wick[::-1] / TICK) * TICK,
        "volume": rng.integers(50, 600, len(idx)),
    }, index=idx)
    return df[df.index.tz_convert(SESSION_TZ).hour != 17]   # CME daily break


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]]


def main(argv):
    df = load_csv(argv[1]) if len(argv) > 1 else synthetic_nq()
    profiles = session_profiles(df)
    # balances are built from completed sessions; the latest session is what we trade
    comps = balance_areas(df, profiles[:-1])
    npocs = naked_pocs(df, profiles[:-1])   # the developing session's POC is not "naked" yet

    print(f"{len(profiles)} sessions, {len(comps)} balance areas, {len(npocs)} naked POCs\n")
    for p in profiles:
        print(f"  {p.end.tz_convert(SESSION_TZ):%d/%m}  POC {p.poc:9.2f}  "
              f"VA {p.val:9.2f}-{p.vah:9.2f}  range {p.low:9.2f}-{p.high:9.2f}")

    all_sigs = []
    for k, c in enumerate(comps):
        hvn, lvn = hvn_lvn(c)
        print(f"\nBalance {c.start.tz_convert(SESSION_TZ) + pd.Timedelta(hours=6):%d/%m}-{c.end.tz_convert(SESSION_TZ):%d/%m}: "
              f"{c.low:.2f}-{c.high:.2f}  POC {c.poc:.2f}  VA {c.val:.2f}-{c.vah:.2f}")
        print(f"  HVN {[round(x) for x in hvn]}\n  LVN {[round(x) for x in lvn]}")
        nxt = comps[k + 1].start if k + 1 < len(comps) else None
        levels = [px for _, px in npocs] + [x for o in comps if o is not c for x in (o.poc, o.high, o.low)]
        sigs = signals(df, c, levels, after=c.end, until=nxt)
        for s in sigs:
            print(f"  {s.time.tz_convert(SESSION_TZ):%d/%m %H:%M}  {s.kind:<14} "
                  f"entry {s.entry:9.2f}  stop {s.stop:9.2f}  target {s.target:9.2f}  R:R {s.rr:.1f}")
        all_sigs += sigs

    print("\nchart ->", plot(df, profiles, comps, npocs, all_sigs))


if __name__ == "__main__":
    main(sys.argv)
