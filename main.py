import os
import json
import logging
import mimetypes
from openai import OpenAI
from dotenv import load_dotenv
from video_transcript import video_transcript
from object_detection import object_detection
from sentiment_analysis import sentiment_analysis
from question_answer import question_answer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_env() -> tuple[str, str]:
    """
    Load and validate API key and video path in the current environment.
    """

    # Load .env
    load_dotenv()

    # Load and validate api key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Check `.env` file or environment variables")

    # Load and validate video path
    video_path = os.getenv("VIDEO_PATH")
    if not video_path:
        raise RuntimeError("Missing VIDEO_PATH. Check `.env` file or environment variables")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check file type
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type or not mime_type.startswith("video/"):
        raise ValueError(f"Unsupported file type: {mime_type}")
    
    return api_key, video_path


def openai_pipeline(api_key: str, video_path: str) -> dict:
    """
    Run the complete video parsing pipeline.
    """
    
    # Initialise the client
    client = OpenAI(api_key=api_key)

    try:
        # Get the complete transcription
        logger.info("Step 1/4: Getting video transcription...")
        transcription = video_transcript(client=client, video_path=video_path, model="whisper-1")

        # Detect objects in the video
        logger.info("Step 2/4: Detecting objects in video...")
        objects = object_detection(client=client, video_path=video_path, model="gpt-4.1", sample_rate=0.5)

        # Analyse the mode and sentiment of the video
        logger.info("Step 3/4: Analysing mode and sentiment...")
        mode_sentiment = sentiment_analysis(client=client, transcription=transcription, model="gpt-4.1")

        # Generate Q&A pairs
        logger.info("Step 4/4: Generating Q&A pairs...")
        qa_pairs = question_answer(client=client, transcription=transcription, model="gpt-4.1")

    except Exception:
        logger.exception("Unexpected error occurred while parsing video")
        raise

    try:
        # Merge and format output
        merge_output = {
            "Transcription": transcription,
            "Objects": json.loads(objects)["objects"],
            "Mode and sentiment": json.loads(mode_sentiment),
            "Q&A pairs": json.loads(qa_pairs)["QA_pairs"]
        }

    except Exception:
        logger.exception("Unexpected error occurred while merging and formatting output")
        raise

    return merge_output


def main():
    """
    Main entry point for execution.
    """
    
    try:
        api_key, video_path = load_env()
        merge_output = openai_pipeline(api_key, video_path)
        
        # Format JSON output
        json_output = json.dumps(merge_output, indent=4, ensure_ascii=False).replace(',\n    "', ',\n\n    "')
        logger.info(f"\n{json_output}")

    except Exception:
        logger.exception("Fatal Error: Execution terminated unexpectedly.")




if __name__ == "__main__":
    main()