import os
import logging
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

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Check `.env` file or environment variables")

# Initialise the client
client = OpenAI(api_key=api_key)

# Load video file
video_path = "AI_Intern_Project.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError("Video file not found")

# Get the complete transcription
transcription = video_transcript(client=client, video_path=video_path, model="whisper-1")
# Detect objects in the video
objects = object_detection(client=client, video_path=video_path, model="gpt-4.1", sample_rate=0.5)
# Analyse the mode and sentiment of the video
mode_sentiment = sentiment_analysis(client=client, transcription=transcription, model="gpt-4.1")
# Generate QA pairs
qa_pairs = question_answer(client=client, transcription=transcription, model="gpt-4.1")








        json_output = json.dumps(
            {"Transcription": transcription.text},
            indent=2
        )


