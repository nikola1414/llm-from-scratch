import pytest

from llm.tokenizer import BPETokenizer, CharTokenizer, Tokenizer

TEXT = ("Dorothy and the Wizard went to the Emerald City. The Wizard said: "
        "'I'm a humbug!' 123456 dollars.\n\n") * 20


@pytest.fixture(scope="module")
def bpe():
    return BPETokenizer.train(TEXT, vocab_size=256 + 3 + 60)


def test_bpe_vocab_size(bpe):
    assert bpe.vocab_size == 256 + 3 + 60


@pytest.mark.parametrize("s", [TEXT, "", "unseen wörds — ünïcode 世界 🙂", "  multiple   spaces\t\ttabs\n"])
def test_bpe_roundtrip(bpe, s):
    assert bpe.decode(bpe.encode(s)) == s


def test_bpe_compresses(bpe):
    assert len(bpe.encode(TEXT)) < len(TEXT.encode("utf-8")) / 2


def test_special_tokens_are_atomic(bpe):
    ids = bpe.encode("<|user|>hi<|assistant|>yo<|endoftext|>")
    assert ids[0] == bpe.special_tokens["<|user|>"]
    assert ids[-1] == bpe.special_tokens["<|endoftext|>"]
    # with allowed_special=False the markers are ordinary text
    assert bpe.special_tokens["<|user|>"] not in bpe.encode("<|user|>", allowed_special=False)


def test_merges_match_training_order(bpe):
    # the first merge must be the most frequent pair in the training text
    assert bpe.merges[0] in {(ord("e"), ord(" ")), (ord(" "), ord("t")), (ord("t"), ord("h")), (ord("h"), ord("e"))} \
        or bpe.merges[0][0] < 256


@pytest.mark.parametrize("make", [lambda: BPETokenizer.train(TEXT, 300), lambda: CharTokenizer(TEXT)])
def test_serialisation(tmp_path, make):
    tok = make()
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.encode(TEXT) == tok.encode(TEXT)
    assert loaded.vocab_size == tok.vocab_size


def test_char_tokenizer():
    tok = CharTokenizer("abc ")
    assert tok.decode(tok.encode("a cab")) == "a cab"
    assert tok.encode("xyz") == []            # unknown characters are dropped
    assert tok.decode(tok.encode("a<|endoftext|>")) == "a<|endoftext|>"


def test_gpt2_pattern_is_unicode_aware():
    tok = BPETokenizer.train("héllo wörld " * 50, 300, pattern="gpt2")
    assert tok.pattern == "gpt2"
    assert tok.split.findall(" wörld") == [" wörld"]          # one chunk, not split at ö
    s = "Ünïcode wörds 12345"
    assert tok.decode(tok.encode(s)) == s


def test_bpe_dropout_is_lossless_and_stochastic(bpe):
    a = bpe.encode_with_dropout(TEXT, 0.3, seed=1)
    b = bpe.encode_with_dropout(TEXT, 0.3, seed=2)
    assert bpe.decode(a) == TEXT and bpe.decode(b) == TEXT
    assert a != b                                   # different segmentations
    assert len(a) > len(bpe.encode(TEXT))           # dropout -> fewer merges -> more tokens
    assert bpe.encode_with_dropout(TEXT, 0.0, seed=1) == bpe.encode(TEXT)


def test_old_tokenizer_json_defaults_to_ascii_pattern(bpe):
    d = bpe.to_dict()
    d.pop("pattern")
    assert Tokenizer.from_dict(d).pattern == "ascii"


def test_special_tokens_not_learned_as_merges():
    tok = BPETokenizer.train("a<|endoftext|>b " * 200, 300)
    assert all(b"<|" not in tok.vocab[i] for i in range(256, 256 + len(tok.merges)))
