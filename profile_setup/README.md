# NQ Volume-Profile Balance Setup

Source: a Deepchart screenshot of NQ (Sep-26 contract, `NQ-202609`) with 30 days of data and 1-tick resolution.

## What the chart shows

| Element | Meaning |
|---|---|
| **Profile per session** (grey/white histograms) | Volume traded at each price for that day. White = inside the 70% value area, blue-grey = outside. |
| **Magenta tick on each profile** | Session **POC**, the price with the most volume. |
| **Long magenta lines across the chart** | **Naked POCs**: old POCs that price has not traded back through yet. They act as magnets and targets. |
| **Yellow bar on the right of each profile** | Session value area (VAL to VAH). |
| **White vertical line** | Session high-to-low range. |
| **Dashed white boxes** | **Balance areas**: several days rotating inside one range, shown as a composite profile. On the chart: about **28,240–29,380** (17–23 Jul) and about **27,300–28,760** (24–30 Jul). |
| **Red bands** | Thin-volume shelves (**LVNs**) or rejection zones above price. Price moves quickly through them or rejects from them, so they act as resistance. Main one: **≈28,720–28,760**, the top of the current balance. |
| **Blue bands** | Heavy-volume shelves (**HVNs**) below price, where buyers defended before. They act as support: ≈28,200, ≈28,000 (composite POC), ≈27,650, ≈27,300. |

## Where price is (30 Jul)

1. The market broke down from the upper balance (28,240–29,380) and built a new one at 27,300–28,760. Its POC is about 28,000, the large profile from 24 Jul.
2. On 29 Jul it tested the balance low (about 27,300) and was rejected. It then rotated through the composite POC.
3. The 30 Jul session opened up near the top of the balance. Its POC is about 28,560, right under the red supply band at 28,720–28,760, which is also the old balance low and LVN.

So price is **at the edge of a balance**, and the setup is a two-way decision:

### A. Fade the edge (rotation, the default in balance)
- **Trigger:** price probes above about 28,760 and fails. A 30-min bar closes back inside the balance, and no acceptance outside has built.
- **Entry:** short on the close back inside.
- **Stop:** above the probe high, with a minimum of about 25 points.
- **Targets:** composite VAH, then the composite POC (about 28,000). A full rotation goes to the balance low (about 27,300).

### B. Breakout and acceptance (initiative)
- **Trigger:** 2 or more 30-min closes above 28,760. Value (the session POC and VA) starts building above the old edge.
- **Entry:** long on acceptance, or on a retest of 28,760 from above.
- **Stop:** back inside the balance.
- **Targets:** first the prior-balance POC or naked POCs around 29,100–29,200, then the prior balance high (about 29,380).

The mirror image applies at the balance low.

## Code: `profile_setup.py`

Pure Python (pandas, numpy, matplotlib). The pipeline:

1. `session_profiles` builds a profile for each CME session (18:00 ET start), with POC/VAH/VAL by the standard 70% expansion from the POC.
2. `naked_pocs` keeps POCs that no later bar has touched.
3. `balance_areas` groups consecutive sessions whose value stays inside a tight composite range (≤ 6% wide, ≥ 3 sessions) and builds composite profiles.
4. `hvn_lvn` finds smoothed composite peaks and troughs, which become the blue and red zones.
5. `signals` scans 30-min bars after each balance:
   - `FADE_SHORT` / `FADE_LONG`: a probe beyond the edge, then a close back inside within 2 bars. The target is the composite POC.
   - `BREAKOUT_LONG` / `BREAKOUT_SHORT`: 2 consecutive closes outside the edge. The target is the next naked POC or prior-balance level at least 1R away, otherwise a half-range measured move.
6. `plot` draws a chart in the Deepchart style, `profile_setup.png`.

```bash
pip install pandas numpy matplotlib
python profile_setup.py                 # demo on synthetic NQ-like data
python profile_setup.py nq_1m.csv       # your data: timestamp,open,high,low,close,volume
```

Tunable constants are at the top of the file: `BIN`, `VALUE_AREA`, `BALANCE_MAX_RANGE_PCT`, `ACCEPT_BARS`, `MIN_PROBE`, `MIN_STOP`, and others.

**Caveats:** Volume is spread evenly across each 1-minute bar's range. Deepchart uses true tick data, so its POCs will differ by a few points. The levels above were read off a screenshot and are approximate. This is a study tool, not a backtested strategy.
