import cv2
import base64
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

def video_to_base64(video_path: str, sample_rate: float=0.5) -> list[str]:
    """
    Convert a video into a list of base64-encoded JPEG images.
    
    Args:
        video_path (str): The path of the video file to be processed.
        sample_rate (float): Number of frames sampled per second (must be > 0).

    Returns:
        list[str]: A list of base64-encoded JPEG images.

    Raises:
        ValueError:  If `sample_rate` is less than or equal to 0.
        RuntimeError:
            - If the video file cannot be opened.
            - If the video metadata is invalid.
            - If an unexpected error occurs while converting a video to base64.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open the video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Validate video metadata
    if fps <= 0 or frame_count <=0:
        raise RuntimeError(f"Invalid video metadata: fps={fps}, frame_count={frame_count}")

    sample_interval = max(1, min(int(fps / sample_rate), int(frame_count)))

    image_count = 0
    base64_images = []

    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break
            if image_count % sample_interval == 0:
                _, buffer = cv2.imencode('.jpg', img)
                base64_str = base64.b64encode(buffer).decode("utf-8")
                base64_images.append(base64_str)
            image_count += 1
    
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred while converting a video to base64: {e}") from e

    finally:
        cap.release()

    return base64_images


def object_detection(client: OpenAI, video_path: str, model: str, sample_rate: float=0.5) -> str:
    """
    Detect distinct objects appearing in a video using OpenAI.
    
    Args:
        client (OpenAI): An initialised OpenAI client with a valid API key.
        video_path (str): The path of the video file to be processed.
        model (str): Model ID used to generate the response, like gpt-4o or o3.
        sample_rate (float): Number of frames sampled per second (must be > 0).
    
    Returns:
        str: A JSON-formatted string containing the detected object.
    
    Raises:
        ValueError:  If `sample_rate` is less than or equal to 0.
        RuntimeError:
            - If frame extraction fails.
            - If an unexpected error occurs while detecting objects.
    """
    
    # Extract frames from the video
    base64_images = video_to_base64(video_path=video_path, sample_rate=sample_rate)

    if not base64_images:
        raise RuntimeError("No frames were extracted from the video")

    # Build input content
    dev_content = [
        {
            "type": "input_text",
            "text": "Return results that strictly match the given JSON format."
        }
    ]

    usr_content = [
        {
            "type": "input_text",
            "text": "List as many distinct objects as possible that appear in these images."
        }
    ]

    for base64_image in base64_images:
        usr_content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "auto"
            }
        )

    # Define JSON schema
    json_schema = {
        "format": {
            "type": "json_schema",
            "name": "object_detection",
            "schema": {
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["objects"],
                "additionalProperties": False
            }
        }
    }

    # Call OpenAI API
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "developer", "content": dev_content},
                {"role": "user", "content": usr_content}
            ],
            text=json_schema
        )
    
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred while detecting objects: {e}") from e
    
    return response.output_text
