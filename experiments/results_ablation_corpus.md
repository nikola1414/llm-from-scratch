| run | data | params | steps | val loss/token | val bits/char | time |
|---|---|---|---|---|---|---|
| g_moe4 | corpus_bpe4k | 2.95M | 2000 | 3.690 | **1.556** | 28.6 min |
| b_muon | corpus_bpe4k | 1.33M | 2000 | 3.737 | **1.576** | 29.0 min |
| e_qknorm_softcap | corpus_bpe4k | 1.33M | 2000 | 3.764 | **1.588** | 28.0 min |
| f_valres_unet | corpus_bpe4k | 1.33M | 2000 | 3.777 | **1.593** | 21.6 min |
| a_base | corpus_bpe4k | 1.33M | 2000 | 3.778 | **1.593** | 28.0 min |
| c_ema | corpus_bpe4k | 1.33M | 2000 | 3.778 | **1.593** | 23.3 min |
| h_gqa2 | corpus_bpe4k | 1.26M | 2000 | 3.797 | **1.601** | 20.5 min |
| d_bpe_dropout | corpus_bpe4k_drop | 1.33M | 2000 | 4.069 | **1.716** | 21.8 min |
