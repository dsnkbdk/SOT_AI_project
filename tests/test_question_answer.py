import pytest
from unittest import mock
import question_answer as qa


def test_question_answer_returns_output_text(monkeypatch):
    class R:
        output_text = '{"QA_pairs":[{"Q":"Q1","A":"A1"}]}'
    class Client:
        class Responses:
            def create(self, **kwargs):
                return R()
        responses = Responses()

    client = Client()
    out = qa.question_answer(client, transcription="hello", model="gpt-4.1")
    assert out == R.output_text


def test_question_answer_raises_on_error(monkeypatch):
    class Client:
        class Responses:
            def create(self, **kwargs):
                raise RuntimeError("boom")
        responses = Responses()

    with pytest.raises(RuntimeError):
        qa.question_answer(Client(), transcription="t", model="gpt-4.1")
