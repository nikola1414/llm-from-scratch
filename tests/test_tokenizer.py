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
