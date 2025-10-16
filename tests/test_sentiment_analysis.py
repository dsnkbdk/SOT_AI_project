import pytest
from unittest import mock
import sentiment_analysis as sa


def test_sentiment_analysis_returns_output_text():
    class R:
        output_text = '{"mode":"m","sentiment":"s","explanation":"e"}'
    class Client:
        class Responses:
            def create(self, **kwargs):
                return R()
        responses = Responses()

    client = Client()
    out = sa.sentiment_analysis(client, transcription="hello", model="gpt-4.1")
    assert out == R.output_text


def test_sentiment_analysis_raises_on_error():
    class Client:
        class Responses:
            def create(self, **kwargs):
                raise RuntimeError("nope")
        responses = Responses()

    with pytest.raises(RuntimeError):
        sa.sentiment_analysis(Client(), transcription="t", model="gpt-4.1")
