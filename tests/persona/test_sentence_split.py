from persona.sentence_split import StreamingSentenceSplitter


def test_emits_complete_sentences_holds_back_trailing_partial():
    s = StreamingSentenceSplitter()
    assert s.feed("Hello there. ") == ["Hello there."]
    assert s.feed("How are you") == []
    assert s.feed("?") == []
    assert s.feed(" Doing well.") == ["How are you?"]
    assert s.flush() == ["Doing well."]


def test_handles_abbreviations_does_not_split_at_dr():
    s = StreamingSentenceSplitter()
    assert s.feed("Dr. Smith said hi.") == []
    assert s.feed(" Then he left.") == ["Dr. Smith said hi."]
    assert s.flush() == ["Then he left."]


def test_handles_ellipsis_as_in_progress_until_disambiguated():
    s = StreamingSentenceSplitter()
    out = s.feed("Well... I think so.")
    # Either the ellipsis is treated as terminating ("Well...") or as part of
    # the same sentence; pysbd treats it as one sentence by default.
    assert out == []
    assert s.feed(" Maybe.") == ["Well... I think so."]
    assert s.flush() == ["Maybe."]


def test_flush_returns_empty_when_buffer_is_whitespace():
    s = StreamingSentenceSplitter()
    s.feed("   \n  ")
    assert s.flush() == []


def test_multiple_sentences_in_one_feed():
    s = StreamingSentenceSplitter()
    out = s.feed("First sentence. Second sentence. Third sentence")
    assert out == ["First sentence.", "Second sentence."]
    assert s.flush() == ["Third sentence"]


def test_empty_feed_returns_nothing():
    s = StreamingSentenceSplitter()
    assert s.feed("") == []
    assert s.flush() == []
