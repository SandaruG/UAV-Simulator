# Exhaustive radio-technology x handover-policy experiment

Study `native-radio-screen` | plan hash `a776ef55f4573f3a` | 12 finished runs | 12 (combo, policy, mode) entries | ranked entries need >= 1 runs | primary metric `blind_time_s` (lower is better)

> Only 0% of runs completed the mission within duration_s=2.0 s (outcome `timeout`). Blind time is then compared over equal flight windows (blind_fraction / wp_progress are recorded); raise scale.duration_s to compare complete missions.

## 1. Ranking by blind_time_s (feasible entries only, top 20)

| rank | combo | policy | video_mode | links | n | blind_time_s mean | sem | 95% CI | P(any blind) | ctrl on-time | ctrl p99 ms | video Q | comm W | comm Wh | Q/W | hard sw | soft sw | completed | pareto | p vs next (Wilcoxon) | p vs best (Holm) | tied with best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cell | bonding | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 41.0801 | 0.2499 | 4.8350 | 0.0027 | 0.0517 | nan | nan | 0.0000 | 1 | nan | nan | 1 |
| 2 | cell | pred_cost | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 41.0801 | 0.2499 | 4.8350 | 0.0027 | 0.0517 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 3 | cell | reactive | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 41.0801 | 0.2499 | 4.8350 | 0.0027 | 0.0517 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 4 | wifi | bonding | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 12.7225 | 0.6249 | 5.2962 | 0.0029 | 0.1180 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 5 | wifi | pred_cost | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 12.7225 | 0.6249 | 5.2962 | 0.0029 | 0.1180 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 6 | wifi | reactive | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 12.7225 | 0.6249 | 5.2962 | 0.0029 | 0.1180 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 7 | wifi+rf24 | reactive | standard | 2 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 11.2728 | 0.6249 | 8.3734 | 0.0047 | 0.0746 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 8 | rf24 | bonding | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 10.7433 | 1.0000 | 9.1487 | 0.0051 | 0.1093 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 9 | rf24 | pred_cost | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 10.7433 | 1.0000 | 9.1487 | 0.0051 | 0.1093 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 10 | rf24 | reactive | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 10.7433 | 1.0000 | 9.1487 | 0.0051 | 0.1093 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 11 | wifi+rf24 | bonding | standard | 2 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 7.7146 | 1.0000 | 9.7014 | 0.0054 | 0.1031 | nan | nan | 0.0000 | 1 | nan | nan | 0 |
| 12 | wifi+rf24 | pred_cost | standard | 2 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 7.7452 | 1.0000 | 10.4184 | 0.0058 | 0.0960 | nan | nan | 0.0000 | 1 | nan | nan | 0 |

`p vs next`: paired Wilcoxon over trials against the next-ranked entry (small p => the ordering of these two is real). `tied with best`: not distinguishable from rank 1 after Holm correction over the top-K.

## 2. Ranking by power efficiency (quality_per_watt, higher is better), top 20

| rank | combo | policy | video_mode | links | n | blind_time_s mean | sem | 95% CI | P(any blind) | ctrl on-time | ctrl p99 ms | video Q | comm W | comm Wh | Q/W | hard sw | soft sw | completed | pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | wifi | bonding | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 12.7225 | 0.6249 | 5.2962 | 0.0029 | 0.1180 | nan | nan | 0.0000 | 1 |
| 2 | wifi | pred_cost | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 12.7225 | 0.6249 | 5.2962 | 0.0029 | 0.1180 | nan | nan | 0.0000 | 1 |
| 3 | wifi | reactive | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 12.7225 | 0.6249 | 5.2962 | 0.0029 | 0.1180 | nan | nan | 0.0000 | 1 |
| 4 | rf24 | bonding | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 10.7433 | 1.0000 | 9.1487 | 0.0051 | 0.1093 | nan | nan | 0.0000 | 1 |
| 5 | rf24 | pred_cost | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 10.7433 | 1.0000 | 9.1487 | 0.0051 | 0.1093 | nan | nan | 0.0000 | 1 |
| 6 | rf24 | reactive | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 10.7433 | 1.0000 | 9.1487 | 0.0051 | 0.1093 | nan | nan | 0.0000 | 1 |
| 7 | wifi+rf24 | bonding | standard | 2 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 7.7146 | 1.0000 | 9.7014 | 0.0054 | 0.1031 | nan | nan | 0.0000 | 1 |
| 8 | wifi+rf24 | pred_cost | standard | 2 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 7.7452 | 1.0000 | 10.4184 | 0.0058 | 0.0960 | nan | nan | 0.0000 | 1 |
| 9 | wifi+rf24 | reactive | standard | 2 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 11.2728 | 0.6249 | 8.3734 | 0.0047 | 0.0746 | nan | nan | 0.0000 | 1 |
| 10 | cell | bonding | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 41.0801 | 0.2499 | 4.8350 | 0.0027 | 0.0517 | nan | nan | 0.0000 | 1 |
| 11 | cell | pred_cost | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 41.0801 | 0.2499 | 4.8350 | 0.0027 | 0.0517 | nan | nan | 0.0000 | 1 |
| 12 | cell | reactive | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 41.0801 | 0.2499 | 4.8350 | 0.0027 | 0.0517 | nan | nan | 0.0000 | 1 |

## 3. Pareto-optimal entries ['blind_time_s', 'comm_power_mean_w', 'video_quality_mean', 'ctrl_lat_p99_ms']

| rank | combo | policy | video_mode | links | n | blind_time_s mean | sem | 95% CI | P(any blind) | ctrl on-time | ctrl p99 ms | video Q | comm W | comm Wh | Q/W | hard sw | soft sw | completed | pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cell | bonding | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 41.0801 | 0.2499 | 4.8350 | 0.0027 | 0.0517 | nan | nan | 0.0000 | 1 |
| 2 | cell | pred_cost | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 41.0801 | 0.2499 | 4.8350 | 0.0027 | 0.0517 | nan | nan | 0.0000 | 1 |
| 3 | cell | reactive | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 41.0801 | 0.2499 | 4.8350 | 0.0027 | 0.0517 | nan | nan | 0.0000 | 1 |
| 4 | wifi | bonding | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 12.7225 | 0.6249 | 5.2962 | 0.0029 | 0.1180 | nan | nan | 0.0000 | 1 |
| 5 | wifi | pred_cost | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 12.7225 | 0.6249 | 5.2962 | 0.0029 | 0.1180 | nan | nan | 0.0000 | 1 |
| 6 | wifi | reactive | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 12.7225 | 0.6249 | 5.2962 | 0.0029 | 0.1180 | nan | nan | 0.0000 | 1 |
| 7 | wifi+rf24 | reactive | standard | 2 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 11.2728 | 0.6249 | 8.3734 | 0.0047 | 0.0746 | nan | nan | 0.0000 | 1 |
| 8 | rf24 | bonding | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 10.7433 | 1.0000 | 9.1487 | 0.0051 | 0.1093 | nan | nan | 0.0000 | 1 |
| 9 | rf24 | pred_cost | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 10.7433 | 1.0000 | 9.1487 | 0.0051 | 0.1093 | nan | nan | 0.0000 | 1 |
| 10 | rf24 | reactive | standard | 1 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 10.7433 | 1.0000 | 9.1487 | 0.0051 | 0.1093 | nan | nan | 0.0000 | 1 |
| 11 | wifi+rf24 | bonding | standard | 2 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 7.7146 | 1.0000 | 9.7014 | 0.0054 | 0.1031 | nan | nan | 0.0000 | 1 |
| 12 | wifi+rf24 | pred_cost | standard | 2 | 1 | 0.0000 | nan | [nan, nan] | 0.0000 | 1.0000 | 7.7452 | 1.0000 | 10.4184 | 0.0058 | 0.0960 | nan | nan | 0.0000 | 1 |

## 4. Best combination per policy

| policy | best combo | n links | blind_time_s | comm W | ctrl p99 ms | video Q |
|---|---|---|---|---|---|---|
| bonding | cell | 1 | 0.0000 | 4.8350 | 41.0801 | 0.2499 |
| pred_cost | cell | 1 | 0.0000 | 4.8350 | 41.0801 | 0.2499 |
| reactive | cell | 1 | 0.0000 | 4.8350 | 41.0801 | 0.2499 |

## 5. Does adding links help? (best feasible entry per number of links)

| links | best combo | policy | blind_time_s | comm W | n entries |
|---|---|---|---|---|---|
| 1 | cell | bonding | 0.0000 | 4.8350 | 9 |
| 2 | wifi+rf24 | reactive | 0.0000 | 8.3734 | 3 |

## 6. Marginal value of each technology on `blind_time_s` (adding it to an already evaluated combo; negative = fewer blind seconds)

| technology | pairs | mean delta | median delta | share improving |
|---|---|---|---|---|
| rf24 | 3 | 0.0000 | 0.0000 | 0.0000 |
| wifi | 3 | 0.0000 | 0.0000 | 0.0000 |

## 7. Policies compared across combinations (paired by combination)

| policy | combos | mean blind | best blind | mean diff vs ref | share of combos better | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| bonding | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| pred_cost | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| reactive | 4 | 0.0000 | 0.0000 | nan | nan | nan |

## 8. How far can you trust small differences?

- Per-run blind time is a multiple of the time step **dt = 0.05 s**; a mean over n = 1 runs therefore has a granularity of dt/n = 0.050000 s.
- Typical standard error of the top-10 means: nan s. Two entries whose CIs overlap are not separable without more seeds.
- Paired resolution needs >= 3 common trials per neighbour pair.
- Values are stored with 9 decimals (fixed); runs.jsonl keeps full float precision. Recording precision is not statistical precision: use the `p vs next` column.
- Re-running any recorded run reproduces it bit-exactly (`audit` command checks the state CRC and every recorded metric).

## 9. Plan, filters and rejections

- Technologies in the universe (3): wifi, rf24, cell
- Strategy `explicit`, max_links 2, 4 combinations, policies ['reactive', 'bonding', 'pred_cost'], scenarios ['jammed_inspection'], video modes ['standard']
- Rejected while generating (static rules): {}
- Rejected by link-budget / tether / coverage pre-screen: {}
