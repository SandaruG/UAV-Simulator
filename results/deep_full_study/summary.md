# Exhaustive radio-technology x handover-policy experiment

Study `deep-full-radio-study` | plan hash `8759223df96ac2e9` | 1640 finished runs | 164 (combo, policy, mode) entries | ranked entries need >= 10 runs | primary metric `blind_time_s` (lower is better)

Notes:
- pairwise: 12 of 136 2-subsets are not coverable under the constraints or were cut by max_combinations

> Only 0% of runs completed the mission within duration_s=120.0 s (outcome `timeout`). Blind time is then compared over equal flight windows (blind_fraction / wp_progress are recorded); raise scale.duration_s to compare complete missions.

## 1. Ranking by blind_time_s (feasible entries only, top 20)

| rank | combo | policy | video_mode | links | n | blind_time_s mean | sem | 95% CI | P(any blind) | ctrl on-time | ctrl p99 ms | video Q | comm W | comm Wh | Q/W | hard sw | soft sw | completed | pareto | p vs next (Wilcoxon) | p vs best (Holm) | tied with best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | elrs24+nr_a+crossfire868 | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9916 | 19.9746 | 0.0042 | 7.9079 | 0.2636 | 0.0005 | nan | nan | 0.0000 | 1 | 1.0000 | nan | 1 |
| 2 | elrs24+nr_a+crossfire868 | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9916 | 19.9746 | 0.0042 | 7.9079 | 0.2636 | 0.0005 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 3 | elrs24+nr_a+crossfire868 | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9916 | 19.9522 | 0.0042 | 7.9830 | 0.2661 | 0.0005 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 4 | elrs24+cell+nr_a | reactive | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9831 | 73.1630 | 0.3626 | 8.1834 | 0.2728 | 0.0434 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 5 | elrs24+cell+nr_a | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9948 | 28.0963 | 0.2816 | 8.2666 | 0.2756 | 0.0331 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 6 | elrs24+cell+nr_a | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9948 | 28.0963 | 0.2816 | 8.2666 | 0.2756 | 0.0331 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 7 | elrs24+cell+nr_a | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9958 | 25.3026 | 0.3495 | 8.3020 | 0.2767 | 0.0411 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 8 | wifi+rf900+nr_a | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9999 | 17.6282 | 0.4309 | 9.4737 | 0.3158 | 0.0453 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 9 | wifi+rf900+nr_a | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9999 | 17.9307 | 0.2849 | 9.7063 | 0.3235 | 0.0292 | nan | nan | 0.0000 | 0 | 1.0000 | 1.0000 | 1 |
| 10 | wifi+rf900+nr_a | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9999 | 17.9307 | 0.2849 | 9.7063 | 0.3235 | 0.0292 | nan | nan | 0.0000 | 0 | 1.0000 | 1.0000 | 1 |
| 11 | rf900+cell+walksnail | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9989 | 23.7003 | 0.3732 | 11.0334 | 0.3678 | 0.0333 | nan | nan | 0.0000 | 0 | 1.0000 | 1.0000 | 1 |
| 12 | rf900+cell+walksnail | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9989 | 23.7003 | 0.3732 | 11.0334 | 0.3678 | 0.0333 | nan | nan | 0.0000 | 0 | 1.0000 | 1.0000 | 1 |
| 13 | wifi+elrs24+lr12g | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9989 | 14.1142 | 0.7639 | 11.0453 | 0.3682 | 0.0685 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 14 | wifi+elrs24+lr12g | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9989 | 14.1142 | 0.7639 | 11.0453 | 0.3682 | 0.0685 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 15 | ocusync+elrs24+rf900 | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9916 | 20.0897 | 0.2317 | 11.5746 | 0.3858 | 0.0191 | nan | nan | 0.0000 | 0 | 1.0000 | 1.0000 | 1 |
| 16 | ocusync+elrs24+rf900 | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9916 | 20.0897 | 0.2317 | 11.5746 | 0.3858 | 0.0191 | nan | nan | 0.0000 | 0 | 1.0000 | 1.0000 | 1 |
| 17 | lr12g+cell+fiber_tether | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 8.9247 | 0.9978 | 12.1573 | 0.4052 | 0.0821 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 18 | lr12g+cell+fiber_tether | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 8.9247 | 0.9978 | 12.1573 | 0.4052 | 0.0821 | nan | nan | 0.0000 | 1 | 1.0000 | 1.0000 | 1 |
| 19 | fpv58_analog+cell+crossfire868 | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9444 | 174.4054 | 0.7019 | 12.2456 | 0.3471 | 0.0572 | nan | nan | 0.0000 | 0 | 1.0000 | 1.0000 | 1 |
| 20 | fpv58_analog+cell+crossfire868 | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9444 | 174.4054 | 0.7019 | 12.2456 | 0.3471 | 0.0572 | nan | nan | 0.0000 | 0 | 1.0000 | 1.0000 | 1 |

`p vs next`: paired Wilcoxon over trials against the next-ranked entry (small p => the ordering of these two is real). `tied with best`: not distinguishable from rank 1 after Holm correction over the top-K.

## 2. Ranking by power efficiency (quality_per_watt, higher is better), top 20

| rank | combo | policy | video_mode | links | n | blind_time_s mean | sem | 95% CI | P(any blind) | ctrl on-time | ctrl p99 ms | video Q | comm W | comm Wh | Q/W | hard sw | soft sw | completed | pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lr12g+cell+fiber_tether | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 8.9247 | 0.9978 | 12.1573 | 0.4052 | 0.0821 | nan | nan | 0.0000 | 1 |
| 2 | lr12g+cell+fiber_tether | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 8.9247 | 0.9978 | 12.1573 | 0.4052 | 0.0821 | nan | nan | 0.0000 | 1 |
| 3 | wifi+rf24+fiber_tether | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 3.9716 | 1.0000 | 13.5573 | 0.4519 | 0.0738 | nan | nan | 0.0000 | 1 |
| 4 | wifi+rf24+fiber_tether | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 3.9716 | 1.0000 | 13.5573 | 0.4519 | 0.0738 | nan | nan | 0.0000 | 1 |
| 5 | wifi+rf24+fiber_tether | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5285 | 1.0000 | 14.1594 | 0.4720 | 0.0706 | nan | nan | 0.0000 | 1 |
| 6 | wifi+elrs24+lr12g | reactive | standard | 3 | 10 | 0.1750 | 0.1750 | [-0.2209, 0.5709] | 0.1000 | 0.9780 | 120.2601 | 0.9346 | 13.2592 | 0.4420 | 0.0704 | nan | nan | 0.0000 | 0 |
| 7 | wifi+elrs24+lr12g | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9989 | 10.6023 | 0.9524 | 13.7240 | 0.4575 | 0.0694 | nan | nan | 0.0000 | 0 |
| 8 | wifi+rf24+fiber_tether | reactive | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5305 | 0.9852 | 14.2705 | 0.4757 | 0.0690 | nan | nan | 0.0000 | 0 |
| 9 | wifi+elrs24+lr12g | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9989 | 14.1142 | 0.7639 | 11.0453 | 0.3682 | 0.0685 | nan | nan | 0.0000 | 1 |
| 10 | wifi+elrs24+lr12g | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9989 | 14.1142 | 0.7639 | 11.0453 | 0.3682 | 0.0685 | nan | nan | 0.0000 | 1 |
| 11 | lr12g+cell+fiber_tether | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5288 | 1.0000 | 14.6515 | 0.4884 | 0.0683 | nan | nan | 0.0000 | 0 |
| 12 | lr12g+cell+fiber_tether | reactive | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5282 | 0.9763 | 14.3455 | 0.4782 | 0.0680 | nan | nan | 0.0000 | 1 |
| 13 | fpv58_analog+elrs24+fiber_tether | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9149 | 3.6231 | 0.9647 | 14.7121 | 0.3843 | 0.0657 | nan | nan | 0.0000 | 0 |
| 14 | fpv58_analog+elrs24+fiber_tether | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9149 | 3.6231 | 0.9647 | 14.7121 | 0.3843 | 0.0657 | nan | nan | 0.0000 | 0 |
| 15 | ocusync+elrs24+rf900 | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9996 | 15.3102 | 0.9205 | 13.9818 | 0.4661 | 0.0657 | nan | nan | 0.0000 | 0 |
| 16 | fpv58_analog+elrs24+fiber_tether | reactive | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5273 | 1.0000 | 15.3880 | 0.5129 | 0.0651 | nan | nan | 0.0000 | 1 |
| 17 | ocusync+elrs24+rf900 | reactive | standard | 3 | 10 | 0.6400 | 0.6400 | [-0.8077, 2.0877] | 0.1000 | 0.9790 | 148.9565 | 0.8985 | 13.6854 | 0.4562 | 0.0650 | nan | nan | 0.0000 | 0 |
| 18 | fpv58_analog+elrs24+fiber_tether | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5264 | 1.0000 | 15.4104 | 0.5137 | 0.0650 | nan | nan | 0.0000 | 1 |
| 19 | rf24+cell+mesh_manet | pred_cost | standard | 3 | 10 | 0.6600 | 0.5044 | [-0.4810, 1.8010] | 0.3000 | 0.9895 | 23.1085 | 0.9297 | 15.1661 | 0.5055 | 0.0612 | nan | nan | 0.0000 | 0 |
| 20 | rf24+cell+mesh_manet | pred_threshold | standard | 3 | 10 | 0.6600 | 0.5044 | [-0.4810, 1.8010] | 0.3000 | 0.9895 | 23.1085 | 0.9297 | 15.1661 | 0.5055 | 0.0612 | nan | nan | 0.0000 | 0 |

## 3. Pareto-optimal entries ['blind_time_s', 'comm_power_mean_w', 'video_quality_mean', 'ctrl_lat_p99_ms']

| rank | combo | policy | video_mode | links | n | blind_time_s mean | sem | 95% CI | P(any blind) | ctrl on-time | ctrl p99 ms | video Q | comm W | comm Wh | Q/W | hard sw | soft sw | completed | pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | elrs24+nr_a+crossfire868 | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9916 | 19.9746 | 0.0042 | 7.9079 | 0.2636 | 0.0005 | nan | nan | 0.0000 | 1 |
| 2 | elrs24+nr_a+crossfire868 | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9916 | 19.9746 | 0.0042 | 7.9079 | 0.2636 | 0.0005 | nan | nan | 0.0000 | 1 |
| 3 | elrs24+nr_a+crossfire868 | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9916 | 19.9522 | 0.0042 | 7.9830 | 0.2661 | 0.0005 | nan | nan | 0.0000 | 1 |
| 4 | elrs24+cell+nr_a | reactive | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9831 | 73.1630 | 0.3626 | 8.1834 | 0.2728 | 0.0434 | nan | nan | 0.0000 | 1 |
| 5 | elrs24+cell+nr_a | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9948 | 28.0963 | 0.2816 | 8.2666 | 0.2756 | 0.0331 | nan | nan | 0.0000 | 1 |
| 6 | elrs24+cell+nr_a | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9948 | 28.0963 | 0.2816 | 8.2666 | 0.2756 | 0.0331 | nan | nan | 0.0000 | 1 |
| 7 | elrs24+cell+nr_a | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9958 | 25.3026 | 0.3495 | 8.3020 | 0.2767 | 0.0411 | nan | nan | 0.0000 | 1 |
| 8 | wifi+rf900+nr_a | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9999 | 17.6282 | 0.4309 | 9.4737 | 0.3158 | 0.0453 | nan | nan | 0.0000 | 1 |
| 13 | wifi+elrs24+lr12g | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9989 | 14.1142 | 0.7639 | 11.0453 | 0.3682 | 0.0685 | nan | nan | 0.0000 | 1 |
| 14 | wifi+elrs24+lr12g | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.9989 | 14.1142 | 0.7639 | 11.0453 | 0.3682 | 0.0685 | nan | nan | 0.0000 | 1 |
| 17 | lr12g+cell+fiber_tether | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 8.9247 | 0.9978 | 12.1573 | 0.4052 | 0.0821 | nan | nan | 0.0000 | 1 |
| 18 | lr12g+cell+fiber_tether | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 8.9247 | 0.9978 | 12.1573 | 0.4052 | 0.0821 | nan | nan | 0.0000 | 1 |
| 22 | wifi+rf24+fiber_tether | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 3.9716 | 1.0000 | 13.5573 | 0.4519 | 0.0738 | nan | nan | 0.0000 | 1 |
| 23 | wifi+rf24+fiber_tether | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 3.9716 | 1.0000 | 13.5573 | 0.4519 | 0.0738 | nan | nan | 0.0000 | 1 |
| 27 | wifi+rf24+fiber_tether | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5285 | 1.0000 | 14.1594 | 0.4720 | 0.0706 | nan | nan | 0.0000 | 1 |
| 29 | lr12g+cell+fiber_tether | reactive | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5282 | 0.9763 | 14.3455 | 0.4782 | 0.0680 | nan | nan | 0.0000 | 1 |
| 36 | fpv58_analog+elrs24+fiber_tether | reactive | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5273 | 1.0000 | 15.3880 | 0.5129 | 0.0651 | nan | nan | 0.0000 | 1 |
| 37 | fpv58_analog+elrs24+fiber_tether | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5264 | 1.0000 | 15.4104 | 0.5137 | 0.0650 | nan | nan | 0.0000 | 1 |
| 63 | rf900+fiber_tether+hdzero | bonding | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1.0000 | 0.5285 | 1.0000 | 22.5583 | 0.7519 | 0.0446 | nan | nan | 0.0000 | 1 |
| 70 | lr12g+nr_a+hdzero | pred_cost | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.6318 | 16.4881 | 1.0000 | 24.0675 | 0.2703 | 0.0417 | nan | nan | 0.0000 | 1 |
| 71 | lr12g+nr_a+hdzero | pred_threshold | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.6318 | 16.4881 | 1.0000 | 24.0675 | 0.2703 | 0.0417 | nan | nan | 0.0000 | 1 |
| 74 | mesh_manet+hdzero+crossfire868 | reactive | standard | 3 | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 27.3613 | 0.2462 | 0.0365 | nan | nan | 0.0000 | 1 |
| 103 | wifi+rf900+nr_a | reactive | standard | 3 | 10 | 0.0250 | 0.0201 | [-0.0204, 0.0704] | 0.2000 | 0.9900 | 33.0394 | 0.4209 | 9.3548 | 0.3118 | 0.0448 | nan | nan | 0.0000 | 1 |
| 137 | elrs24+nr_a+crossfire868 | reactive | standard | 3 | 10 | 0.9800 | 0.8134 | [-0.8598, 2.8198] | 0.2000 | 0.9743 | 184.5914 | 0.0042 | 7.3775 | 0.2459 | 0.0006 | nan | nan | 0.0000 | 1 |

## 4. Best combination per policy

| policy | best combo | n links | blind_time_s | comm W | ctrl p99 ms | video Q |
|---|---|---|---|---|---|---|
| bonding | elrs24+nr_a+crossfire868 | 3 | 0.0000 | 7.9830 | 19.9522 | 0.0042 |
| pred_cost | elrs24+nr_a+crossfire868 | 3 | 0.0000 | 7.9079 | 19.9746 | 0.0042 |
| pred_threshold | elrs24+nr_a+crossfire868 | 3 | 0.0000 | 7.9079 | 19.9746 | 0.0042 |
| reactive | elrs24+cell+nr_a | 3 | 0.0000 | 8.1834 | 73.1630 | 0.3626 |

## 5. Does adding links help? (best feasible entry per number of links)

| links | best combo | policy | blind_time_s | comm W | n entries |
|---|---|---|---|---|---|
| 2 | rf24+hdzero | pred_cost | 1.9250 | 20.2846 | 8 |
| 3 | elrs24+nr_a+crossfire868 | pred_cost | 0.0000 | 7.9079 | 156 |

## 7. Policies compared across combinations (paired by combination)

| policy | combos | mean blind | best blind | mean diff vs ref | share of combos better | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| bonding | 41 | 0.7173 | 0.0000 | -0.1930 | 0.3659 | 0.0289 |
| pred_cost | 41 | 0.6227 | 0.0000 | -0.2877 | 0.3902 | 0.0383 |
| pred_threshold | 41 | 0.6227 | 0.0000 | -0.2877 | 0.3902 | 0.0383 |
| reactive | 41 | 0.9104 | 0.0000 | nan | nan | nan |

## 8. How far can you trust small differences?

- Per-run blind time is a multiple of the time step **dt = 0.05 s**; a mean over n = 10 runs therefore has a granularity of dt/n = 0.005000 s.
- Typical standard error of the top-10 means: 0.000000 s. Two entries whose CIs overlap are not separable without more seeds.
- Because every entry uses the same trial seeds (common random numbers), neighbours are compared on paired differences; the median SD of those differences in the top-10 is 0.000000 s, so a paired difference of about 0.000000 s is the smallest effect resolvable with n = 10 (alpha 0.05, 50% power).
- Values are stored with 6 decimals (fixed); runs.jsonl keeps full float precision. Recording precision is not statistical precision: use the `p vs next` column.
- Re-running any recorded run reproduces it bit-exactly (`audit` command checks the state CRC and every recorded metric).

## 9. Plan, filters and rejections

- Technologies in the universe (16): fpv58_analog, ocusync, wifi, elrs24, rf24, lr12g, rf900, cell, nr_a, sim_bond, leo, mesh_manet, fiber_tether, walksnail, hdzero, crossfire868
- Strategy `pairwise`, max_links 3, 41 combinations, policies ['reactive', 'bonding', 'pred_cost', 'pred_threshold'], scenarios ['jammed_inspection', 'urban_canyon'], video modes ['standard']
- Rejected while generating (static rules): {'exclusive_or_conflict': 329, 'missing_forced_include_any': 244, 'no_realtime_video_link': 32}
- Rejected by link-budget / tether / coverage pre-screen: {'tether_too_short_or_no_control_coverage': 2}
- pairwise: 12 of 136 2-subsets are not coverable under the constraints or were cut by max_combinations
