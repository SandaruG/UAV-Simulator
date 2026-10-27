# Exhaustive radio-technology x handover-policy experiment

Study `deep-full-radio-study` | plan hash `31a3ca16ba4c0128` | 2 finished runs | 1 (combo, policy, mode) entries | ranked entries need >= 2 runs | primary metric `blind_time_s` (lower is better)

## 1. Ranking by blind_time_s (feasible entries only, top 20)

| rank | combo | policy | video_mode | links | n | blind_time_s mean | sem | 95% CI | P(any blind) | ctrl on-time | ctrl p99 ms | video Q | comm W | comm Wh | Q/W | hard sw | soft sw | completed | pareto | p vs next (Wilcoxon) | p vs best (Holm) | tied with best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | elrs24+nr_a+crossfire868 | pred_cost | standard | 3 | 2 | 40.5250 | 40.5250 | [-474.3856, 555.4356] | 0.5000 | 0.9272 | 240.4596 | 0.0008 | 7.9347 | 1.3225 | 0.0001 | nan | nan | 1.0000 | 1 | nan | nan | 1 |

`p vs next`: paired Wilcoxon over trials against the next-ranked entry (small p => the ordering of these two is real). `tied with best`: not distinguishable from rank 1 after Holm correction over the top-K.

## 2. Ranking by power efficiency (quality_per_watt, higher is better), top 20

| rank | combo | policy | video_mode | links | n | blind_time_s mean | sem | 95% CI | P(any blind) | ctrl on-time | ctrl p99 ms | video Q | comm W | comm Wh | Q/W | hard sw | soft sw | completed | pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | elrs24+nr_a+crossfire868 | pred_cost | standard | 3 | 2 | 40.5250 | 40.5250 | [-474.3856, 555.4356] | 0.5000 | 0.9272 | 240.4596 | 0.0008 | 7.9347 | 1.3225 | 0.0001 | nan | nan | 1.0000 | 1 |

## 3. Pareto-optimal entries ['blind_time_s', 'comm_power_mean_w', 'video_quality_mean', 'ctrl_lat_p99_ms']

| rank | combo | policy | video_mode | links | n | blind_time_s mean | sem | 95% CI | P(any blind) | ctrl on-time | ctrl p99 ms | video Q | comm W | comm Wh | Q/W | hard sw | soft sw | completed | pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | elrs24+nr_a+crossfire868 | pred_cost | standard | 3 | 2 | 40.5250 | 40.5250 | [-474.3856, 555.4356] | 0.5000 | 0.9272 | 240.4596 | 0.0008 | 7.9347 | 1.3225 | 0.0001 | nan | nan | 1.0000 | 1 |

## 4. Best combination per policy

| policy | best combo | n links | blind_time_s | comm W | ctrl p99 ms | video Q |
|---|---|---|---|---|---|---|
| pred_cost | elrs24+nr_a+crossfire868 | 3 | 40.5250 | 7.9347 | 240.4596 | 0.0008 |

## 5. Does adding links help? (best feasible entry per number of links)

| links | best combo | policy | blind_time_s | comm W | n entries |
|---|---|---|---|---|---|
| 3 | elrs24+nr_a+crossfire868 | pred_cost | 40.5250 | 7.9347 | 1 |

## 7. Policies compared across combinations (paired by combination)

| policy | combos | mean blind | best blind | mean diff vs ref | share of combos better | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| pred_cost | 1 | 40.5250 | 40.5250 | nan | nan | nan |

## 8. How far can you trust small differences?

- Per-run blind time is a multiple of the time step **dt = 0.05 s**; a mean over n = 2 runs therefore has a granularity of dt/n = 0.025000 s.
- Typical standard error of the top-10 means: 40.525000 s. Two entries whose CIs overlap are not separable without more seeds.
- Paired resolution needs >= 3 common trials per neighbour pair.
- Values are stored with 6 decimals (fixed); runs.jsonl keeps full float precision. Recording precision is not statistical precision: use the `p vs next` column.
- Re-running any recorded run reproduces it bit-exactly (`audit` command checks the state CRC and every recorded metric).

## 9. Plan, filters and rejections

- Technologies in the universe (16): fpv58_analog, ocusync, wifi, elrs24, rf24, lr12g, rf900, cell, nr_a, sim_bond, leo, mesh_manet, fiber_tether, walksnail, hdzero, crossfire868
- Strategy `explicit`, max_links 3, 1 combinations, policies ['pred_cost'], scenarios ['jammed_inspection', 'urban_canyon'], video modes ['standard']
- Rejected while generating (static rules): {}
- Rejected by link-budget / tether / coverage pre-screen: {}
