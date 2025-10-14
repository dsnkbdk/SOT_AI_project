import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

def video_transcript(
    client: OpenAI,
    video_path: str,
    model: str,
    language: str="en",
    response_format: str="json"
) -> str:
    """
    Transcribe a video/audio file using the OpenAI API and return the result in JSON format.
    
    Args:
        client (OpenAI): An initialised OpenAI client with a valid API key.
        video_path (str): The local path to the video/audio file to be transcribed.
        model (str): ID of the model to use. The options are gpt-4o-transcribe, gpt-4o-mini-transcribe, and whisper-1.
        language (str, optional): Supplying the input language in ISO-639-1 format will improve accuracy and latency.
        response_format (str, optional): The format of the output, (e.g., json, text, srt, verbose_json, or vtt).
    
    Returns:
        str: A JSON-formatted string containing the transcription text.
    """
    try:
        with open(video_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model=model,
                language=language,
                response_format=response_format
            )
        return json.dumps(
            {"Transcription": transcription.text},
            indent=2
        )

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return json.dumps(
            {"status": "Transcription failed", "message": str(e)},
            indent=2
        )
