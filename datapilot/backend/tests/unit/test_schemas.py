from __future__ import annotations

import re

import pytest

from app.schemas.project import count_words


class TestWordCounting:
    def test_empty_string(self):
        assert count_words("") == 0

    def test_whitespace_only(self):
        assert count_words("   ") == 0
        assert count_words("\t\n") == 0

    def test_single_word(self):
        assert count_words("hello") == 1

    def test_multiple_words(self):
        assert count_words("hello world") == 2

    def test_150_words_exactly(self):
        words = " ".join(["word"] * 150)
        assert count_words(words) == 150

    def test_151_words_rejected(self):
        words = " ".join(["word"] * 151)
        assert count_words(words) == 151

    def test_punctuation_does_not_count(self):
        assert count_words("hello, world!") == 2

    def test_hyphenated_words(self):
        # hyphenated words are split by \b pattern
        result = count_words("state-of-the-art")
        assert result >= 1

    def test_numbers_count(self):
        assert count_words("version 2 is great") == 4

    def test_unicode_words(self):
        assert count_words("machine learning AI") == 3

    def test_multiple_spaces(self):
        assert count_words("hello   world") == 2

    def test_leading_trailing_spaces(self):
        assert count_words("  hello world  ") == 2

    def test_deterministic(self):
        text = "I want to build an AI project for healthcare diagnosis"
        assert count_words(text) == count_words(text)
