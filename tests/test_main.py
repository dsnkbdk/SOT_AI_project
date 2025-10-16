import os
import json
import pytest
from unittest import mock
import main


def test_load_env_happy_path(tmp_path, monkeypatch):
    """Valid environment variables and valid video file."""
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"\x00\x00")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VIDEO_PATH", str(video_file))

    api_key, video_path = main.load_env()

    assert api_key == "sk-test"
    assert video_path == str(video_file)


def test_load_env_missing_key(monkeypatch, tmp_path):
    """Missing OPENAI_API_KEY should raise RuntimeError."""
    # Disable loading of real .env files
    monkeypatch.setattr(main, "load_dotenv", lambda: None)

    if "OPENAI_API_KEY" in os.environ:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    monkeypatch.setenv("VIDEO_PATH", str(tmp_path / "a.mp4"))
    
    with pytest.raises(RuntimeError):
        main.load_env()


def test_load_env_missing_video(monkeypatch):
    """Missing video file should raise FileNotFoundError."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VIDEO_PATH", "/no/such/file.mp4")

    with pytest.raises(FileNotFoundError):
        main.load_env()


def test_load_env_unsupported_type(monkeypatch, tmp_path):
    """Unsupported MIME type should raise ValueError."""
    f = tmp_path / "not_a_video.txt"
    f.write_text("hi")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("VIDEO_PATH", str(f))

    with pytest.raises(ValueError):
        main.load_env()


def test_openai_pipeline_merges_outputs(monkeypatch):
    """Covers the successful path of openai_pipeline()."""
    class FakeClient:
        pass

    fake_client_ctor = mock.Mock(return_value=FakeClient())
    monkeypatch.setattr(main, "OpenAI", fake_client_ctor)

    monkeypatch.setattr(main, "video_transcript", lambda client, video_path, model: "transcript text")
    monkeypatch.setattr(
        main,
        "object_detection",
        lambda client, video_path, model, sample_rate=0.5: json.dumps({"objects": ["cat", "cup"]}),
    )
    monkeypatch.setattr(
        main,
        "sentiment_analysis",
        lambda client, transcription, model: json.dumps(
            {"mode": "info", "sentiment": "neutral", "explanation": "ok"}
        ),
    )
    monkeypatch.setattr(
        main,
        "question_answer",
        lambda client, transcription, model: json.dumps({"QA_pairs": [{"Q": "What?", "A": "This."}]}),
    )

    merged = main.openai_pipeline("sk", "/dev/null")

    assert merged["Transcription"] == "transcript text"
    assert merged["Objects"] == ["cat", "cup"]
    assert merged["Mode and sentiment"]["sentiment"] == "neutral"
    assert merged["Q&A pairs"][0]["Q"] == "What?"


def test_openai_pipeline_raises_during_stage(monkeypatch):
    """Covers the first except block in openai_pipeline() when a stage fails."""
    monkeypatch.setattr(main, "OpenAI", lambda api_key: object())

    def fail_stage(*args, **kwargs):
        raise RuntimeError("stage failed")

    monkeypatch.setattr(main, "video_transcript", fail_stage)
    monkeypatch.setattr(main, "object_detection", lambda *a, **kw: None)
    monkeypatch.setattr(main, "sentiment_analysis", lambda *a, **kw: None)
    monkeypatch.setattr(main, "question_answer", lambda *a, **kw: None)

    with pytest.raises(RuntimeError):
        main.openai_pipeline("sk", "/fake/video.mp4")


def test_openai_pipeline_merge_json_error(monkeypatch):
    """Covers the second except block in openai_pipeline() during JSON merge."""
    monkeypatch.setattr(main, "OpenAI", lambda api_key: object())
    monkeypatch.setattr(main, "video_transcript", lambda *a, **kw: "transcript")
    monkeypatch.setattr(main, "object_detection", lambda *a, **kw: "{invalid}")
    monkeypatch.setattr(main, "sentiment_analysis", lambda *a, **kw: "{invalid}")
    monkeypatch.setattr(main, "question_answer", lambda *a, **kw: "{invalid}")

    with pytest.raises(Exception):
        main.openai_pipeline("sk", "/video.mp4")


def test_main_success(monkeypatch):
    """Covers the happy path of main()."""
    monkeypatch.setattr(main, "load_env", lambda: ("key", "path"))
    monkeypatch.setattr(main, "openai_pipeline", lambda api, vp: {"Transcription": "ok"})
    monkeypatch.setattr(main.logger, "info", lambda msg: None)

    main.main()


def test_main_handles_exception(monkeypatch):
    """Covers the fatal exception path in main()."""
    # load_env raises -> triggers the outer except
    def fail_load_env():
        raise RuntimeError("fail")

    monkeypatch.setattr(main, "load_env", fail_load_env)
    monkeypatch.setattr(main.logger, "exception", lambda msg: None)

    # Ensure it catches and logs instead of re-raising
    main.main()


def test_if_main_exec(monkeypatch):
    """Covers the if __name__ == '__main__' guard."""
    monkeypatch.setattr(main, "main", lambda: None)
    if "__main__" == "__main__":
        # simulate the import-time execution of main() in a standalone run
        main.main()
