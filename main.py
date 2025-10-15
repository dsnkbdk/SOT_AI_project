import os
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

# Load .env
load_dotenv()

# Load api key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Check `.env` file or environment variables")

# Load video path
video_path = os.getenv("VIDEO_PATH")
if not video_path:
    raise RuntimeError("VIDEO_PATH is missing. Check `.env` file or environment variables")
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

# Check file type
mime_type, _ = mimetypes.guess_type(video_path)
if mime_type.startswith("video/"):
    # Initialise the client
    client = OpenAI(api_key=api_key)

# Get the complete transcription
transcription = video_transcript(client=client, video_path=video_path, model="whisper-1")
# Detect objects in the video
objects = object_detection(client=client, video_path=video_path, model="gpt-4.1", sample_rate=0.5)
# Analyse the mode and sentiment of the video
mode_sentiment = sentiment_analysis(client=client, transcription=transcription, model="gpt-4.1")
# Generate QA pairs
qa_pairs = question_answer(client=client, transcription=transcription, model="gpt-4.1")

# Format JSON output
merge_output = {
    "Transcription": transcription,
    "Objects": json.loads(objects)["objects"],
    "Mode and sentiment": json.loads(mode_sentiment),
    "QA pairs": json.loads(qa_pairs)["QA_pairs"]
}
json_output = json.dumps(merge_output, indent=4, ensure_ascii=False).replace(',\n    "', ',\n\n    "')
logger.info(json_output)

