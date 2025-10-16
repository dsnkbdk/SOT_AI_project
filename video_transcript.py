import os
import logging
import tempfile
from openai import OpenAI
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)

def video_transcript(client: OpenAI, video_path: str, model: str, language: str="en") -> str:
    """
    Transcribe a video file using the OpenAI API.
    The function extracts the audio track before sending it to OpenAI.
    
    Args:
        client (OpenAI): An initialised OpenAI client with a valid API key.
        video_path (str): The path of the video file to be processed.
        model (str): ID of the model to use. The options are gpt-4o-transcribe, gpt-4o-mini-transcribe, and whisper-1.
        language (str, optional): Supplying the input language in ISO-639-1 format will improve accuracy and latency.
    
    Returns:
        str: The complete transcription of the video.
    
    Raises:
        RuntimeError: If an unexpected error occurs while transcribing.
    """
    
    try:
        # Extract audio track
        logger.info("Extracting audio track...")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            with VideoFileClip(video_path) as clip:
                clip.audio.write_audiofile(temp_audio.name, logger=None)
            audio_file = temp_audio.name

        # Call OpenAI API
        logger.info("Transcribing video...")
        with open(audio_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model=model,
                language=language,
                response_format="json",
                prompt="Transcribe exactly what is spoken. Ignore any background music or noise that may be present."
            )
    
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred while transcribing") from e
    
    finally:
        if os.path.exists(temp_audio.name):
            os.remove(temp_audio.name)
    
    return transcription.text
    