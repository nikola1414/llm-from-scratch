| run | data | params | steps | val loss/token | val bits/char | time |
|---|---|---|---|---|---|---|
| bpe512_reg | oz_bpe512 | 0.87M | 1500 | 2.876 | **2.034** | 12.6 min |
| bpe512_small_reg (early-stopped) | oz_bpe512 | 2.46M | 1101 | 2.935 | **2.075** | 22.8 min |
| char_small (stopped, best.pt) | oz_char | 1.87M | 1500 | 1.431 | **2.064** | 25 min |
