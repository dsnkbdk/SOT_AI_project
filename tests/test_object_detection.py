import base64
import pytest
from unittest import mock
import object_detection as od


def make_fake_frame():
    # A tiny fake JPEG buffer
    return b"\xff\xd8\xff\xdb\x00\x43\x00"


def build_cap_mock(is_open=True, fps=10, frame_count=20, read_frames=20):
    """Helper to build a mocked VideoCapture with controlled metadata and frames."""
    cap_mock = mock.Mock()
    cap_mock.isOpened.return_value = is_open
    cap_mock.get.side_effect = [fps, frame_count]

    # Create N True reads followed by False to terminate
    frames = [True] * read_frames + [False]
    imgs = [mock.sentinel.img] * read_frames

    def read_gen():
        for ok, img in zip(frames, imgs + [None]):
            yield (ok, img)

    it = read_gen()
    cap_mock.read.side_effect = lambda: next(it)
    return cap_mock


def test_video_to_base64_happy_path(monkeypatch):
    cap_mock = build_cap_mock(is_open=True, fps=10, frame_count=20, read_frames=20)
    monkeypatch.setattr(od.cv2, "VideoCapture", mock.Mock(return_value=cap_mock))
    monkeypatch.setattr(od.cv2, "imencode", mock.Mock(return_value=(True, make_fake_frame())))

    out = od.video_to_base64("whatever.mp4", sample_rate=0.5)
    assert isinstance(out, list)
    assert len(out) >= 1
    base64.b64decode(out[0])


# Input validation & metadata
def test_video_to_base64_sample_rate_validation():
    with pytest.raises(ValueError):
        od.video_to_base64("x.mp4", sample_rate=0)


def test_video_to_base64_cannot_open(monkeypatch):
    cap_mock = build_cap_mock(is_open=False)
    monkeypatch.setattr(od.cv2, "VideoCapture", mock.Mock(return_value=cap_mock))
    with pytest.raises(RuntimeError):
        od.video_to_base64("bad.mp4", sample_rate=1.0)


def test_video_to_base64_invalid_metadata(monkeypatch):
    """Covers fps<=0 or frame_count<=0 metadata validation path."""
    cap_mock = build_cap_mock(is_open=True, fps=0, frame_count=10, read_frames=0)
    monkeypatch.setattr(od.cv2, "VideoCapture", mock.Mock(return_value=cap_mock))
    with pytest.raises(RuntimeError):
        od.video_to_base64("bad_meta.mp4", sample_rate=1.0)


# Warning branches (interval)
def test_video_to_base64_warns_sample_too_low(monkeypatch, caplog):
    """sample_interval > frame_count: only one frame will be sampled."""
    # fps=10, frame_count=5, sample_rate=0.1 => int(10/0.1)=100 > 5
    cap_mock = build_cap_mock(is_open=True, fps=10, frame_count=5, read_frames=5)
    monkeypatch.setattr(od.cv2, "VideoCapture", mock.Mock(return_value=cap_mock))
    monkeypatch.setattr(od.cv2, "imencode", mock.Mock(return_value=(True, make_fake_frame())))

    with caplog.at_level("WARNING"):
        out = od.video_to_base64("v.mp4", sample_rate=0.1)
    assert len(out) >= 1
    assert any("only one frame will be sampled" in m for m in caplog.messages)


def test_video_to_base64_warns_sample_too_high(monkeypatch, caplog):
    """sample_interval < 1: all frames will be sampled (may exceed limit)."""
    # fps=10, sample_rate=1000 => int(10/1000)=0 < 1
    cap_mock = build_cap_mock(is_open=True, fps=10, frame_count=3, read_frames=3)
    monkeypatch.setattr(od.cv2, "VideoCapture", mock.Mock(return_value=cap_mock))
    monkeypatch.setattr(od.cv2, "imencode", mock.Mock(return_value=(True, make_fake_frame())))

    with caplog.at_level("WARNING"):
        out = od.video_to_base64("v.mp4", sample_rate=1000)
    assert len(out) == 3
    assert any("all frames will be sampled" in m for m in caplog.messages)


# Exception inside processing
def test_video_to_base64_imencode_failure_raises(monkeypatch):
    """cv2.imencode returns unusable buffer -> triggers except -> RuntimeError."""
    cap_mock = build_cap_mock(is_open=True, fps=10, frame_count=2, read_frames=2)
    monkeypatch.setattr(od.cv2, "VideoCapture", mock.Mock(return_value=cap_mock))

    # Return (False, None) so base64.b64encode(None) raises TypeError caught by except
    monkeypatch.setattr(od.cv2, "imencode", mock.Mock(return_value=(False, None)))

    with pytest.raises(RuntimeError):
        od.video_to_base64("broken.mp4", sample_rate=1.0)


# object_detection() branches
def test_object_detection_calls_openai_and_returns_text(monkeypatch):
    # Avoid real frame extraction
    monkeypatch.setattr(od, "video_to_base64", lambda video_path, sample_rate: ["ZmFrZQ=="])

    class R:
        output_text = '{"objects":["cat","tree"]}'
    class Client:
        class Responses:
            def create(self, **kwargs):
                return R()
        responses = Responses()

    client = Client()
    result = od.object_detection(client=client, video_path="x.mp4", model="gpt-4.1", sample_rate=1.0)
    assert result == '{"objects":["cat","tree"]}'


def test_object_detection_no_frames_raises(monkeypatch):
    """Covers 'No frames were extracted from the video' branch."""
    monkeypatch.setattr(od, "video_to_base64", lambda *a, **kw: [])
    class Client:
        responses = mock.Mock()
    with pytest.raises(RuntimeError):
        od.object_detection(Client(), video_path="x.mp4", model="gpt-4.1", sample_rate=1.0)


def test_object_detection_openai_error_wrapped(monkeypatch):
    """Covers try/except around client.responses.create (wrapped as RuntimeError)."""
    monkeypatch.setattr(od, "video_to_base64", lambda *a, **kw: ["ZmFrZQ=="])

    class Client:
        class Responses:
            def create(self, **kwargs):
                raise RuntimeError("API fail")
        responses = Responses()

    with pytest.raises(RuntimeError):
        od.object_detection(Client(), video_path="x.mp4", model="gpt-4.1", sample_rate=1.0)
