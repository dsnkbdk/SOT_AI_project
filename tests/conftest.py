import pytest
from types import SimpleNamespace


@pytest.fixture
def mock_openai_client():
    class _Client:
        def __init__(self):
            self.audio = SimpleNamespace(
                transcriptions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(text="hello world transcription")
                )
            )
            self.responses = SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(output_text='{"objects": ["car", "tree"]}')
            )
    return _Client()


@pytest.fixture
def mock_openai_client_sentiment():
    class _Client:
        def __init__(self):
            self.responses = SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    output_text='{"mode":"educational","sentiment":"positive","explanation":"clear"}'
                )
            )
    return _Client()


@pytest.fixture
def mock_openai_client_qa():
    class _Client:
        def __init__(self):
            self.responses = SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    output_text='{"QA_pairs":[{"Q":"q1","A":"a1"},{"Q":"q2","A":"a2"}]}'
                )
            )
    return _Client()
