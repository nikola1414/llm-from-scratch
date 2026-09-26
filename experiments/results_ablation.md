| run | data | params | steps | val loss/token | val bits/char | time |
|---|---|---|---|---|---|---|
| 3_v2_arch_char | oz_char | 0.81M | 2000 | 1.435 | **2.070** | 14.1 min |
| 2_v1_arch_v2_recipe | oz_char | 0.83M | 2000 | 1.465 | **2.113** | 12.7 min |
| 4_v2_bpe512 | oz_bpe512 | 0.87M | 2000 | 3.004 | **2.124** | 14.5 min |
| 5_v2_bpe1024 | oz_bpe1024 | 0.94M | 2000 | 3.823 | **2.130** | 15.5 min |
| 6_v2_bpe2048 | oz_bpe2048 | 1.07M | 2000 | 4.489 | **2.133** | 17.6 min |
| 1_v1_baseline | oz_char | 0.82M | 2000 | 1.685 | **2.431** | 6.3 min |
