"""llm — v2 of the from-scratch GPT: a modern, efficient, well-tested implementation.

Modules
    tokenizer   byte-level BPE (trained from scratch) and character tokenizers
    model       GPT with RoPE, RMSNorm, SwiGLU, grouped-query attention, fused
                scaled-dot-product attention, weight tying and a KV cache
    data        tokenize text into binary token files; memory-mapped batch loader
    prepare     CLI: train a tokenizer and build train.bin / val.bin
    train       CLI: pretraining loop (AdamW, warmup+cosine LR, grad accumulation,
                gradient clipping, mixed precision, torch.compile, checkpoints)
    finetune    CLI: supervised instruction finetuning with prompt-masked loss
    evaluate    CLI: full validation loss, perplexity and bits-per-character
    generate    CLI: sampling / interactive chat with streaming output
"""
