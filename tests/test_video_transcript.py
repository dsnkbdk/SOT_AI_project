import io
import os
import pytest
from unittest import mock
import video_transcript as vt


def test_video_transcript_happy_path(monkeypatch, tmp_path):
    # Create a fake temp audio path the function would use
    # We will intercept moviepy and the OpenAI call.

    # Mock VideoFileClip to write audio to a temp file
    class FakeAudio:
        def write_audiofile(self, path, logger=None):
            with open(path, "wb") as f:
                f.write(b"FAKEAUDIO")

    class FakeClip:
        audio = FakeAudio()
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass

    monkeypatch.setattr(vt, "VideoFileClip", mock.Mock(return_value=FakeClip()))

    # Mock client.audio.transcriptions.create to return object with "text"
    class R:
        text = "hello world"
    class Client:
        class Audio:
            class Transcriptions:
                def create(self, **kwargs):
                    return R()
            transcriptions = Transcriptions()
        audio = Audio()

    # Spy on os.remove to verify cleanup
    remove_calls = []
    def fake_remove(p):
        remove_calls.append(p)
        if os.path.exists(p):
            os.unlink(p)
    monkeypatch.setattr(vt.os, "remove", fake_remove)

    # Act
    out = vt.video_transcript(Client(), video_path="video.mp4", model="whisper-1")

    # Assert
    assert out == "hello world"
    # Ensure temp file was cleaned up
    assert len(remove_calls) == 1


def test_video_transcript_raises_on_error(monkeypatch):
    # Make VideoFileClip raise
    def bad_vfc(*args, **kwargs):
        raise RuntimeError("decode error")
    monkeypatch.setattr(vt, "VideoFileClip", bad_vfc)

    class Client:
        pass

    with pytest.raises(RuntimeError):
        vt.video_transcript(Client(), video_path="x.mp4", model="whisper-1")
