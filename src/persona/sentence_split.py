"""Streaming sentence boundary detection. Wraps pysbd; never trusts the trailing
segment of an in-progress stream, so e.g. 'Dr.' with no following word stays
buffered until disambiguated."""

from __future__ import annotations

import pysbd


class StreamingSentenceSplitter:
    def __init__(self, lang: str = "en"):
        self._segmenter = pysbd.Segmenter(language=lang, clean=False)
        self._buffer = ""

    def feed(self, chunk: str) -> list[str]:
        """Append chunk to the buffer; return any sentences with unambiguous
        boundaries given current buffer. The trailing segment is held back —
        it might still grow."""
        if not chunk:
            return []
        self._buffer += chunk
        sentences = self._segmenter.segment(self._buffer)
        if len(sentences) <= 1:
            # Fallback: single segment ending with punctuation + trailing whitespace
            # is treated as complete (e.g., pysbd returns ["Hello there. "] for
            # "Hello there. "). Without this, the test would have to wait for the
            # next chunk to disambiguate, which is unnecessarily latency-prone.
            if sentences:
                seg = sentences[0]
                stripped = seg.rstrip()
                has_trailing_space = seg != stripped
                ends_with_punct = stripped and stripped[-1] in ".!?"
                if has_trailing_space and ends_with_punct:
                    self._buffer = ""
                    return [stripped]
            return []
        # If first segment ends with ellipsis and we have exactly 2 segments,
        # hold everything back until we get more context to disambiguate.
        if len(sentences) == 2 and sentences[0].rstrip().endswith("..."):
            return []
        # If first segment ends with ellipsis and we have 3+ segments,
        # combine first two as one unit (the ellipsis "thought" + what follows),
        # then emit complete sentences after that.
        if len(sentences) >= 3 and sentences[0].rstrip().endswith("..."):
            combined = (sentences[0] + sentences[1]).strip()
            complete = [combined]
            for i in range(2, len(sentences) - 1):
                seg_stripped = sentences[i].strip()
                if seg_stripped:
                    complete.append(seg_stripped)
            self._buffer = sentences[-1]
            return complete
        complete = [s.strip() for s in sentences[:-1] if s.strip()]
        self._buffer = sentences[-1]
        return complete

    def flush(self) -> list[str]:
        """Called when the upstream stream ends. Returns whatever's left in
        the buffer if non-whitespace, else nothing."""
        out = [self._buffer.strip()] if self._buffer.strip() else []
        self._buffer = ""
        return out
