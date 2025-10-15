import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

def sentiment_analysis(client: OpenAI, transcription: str, model: str) -> str:
    """
    Analyse the overall mode and sentiment of the video using OpenAI.
    
    Args:
        client (OpenAI): An initialised OpenAI client with a valid API key.
        transcription (str): The complete transcription of the video.
        model (str): Model ID used to generate the response, like gpt-4o or o3.
    
    Returns:
        str: A JSON-formatted string containing the mode and sentiment analysis results.
    
    Raises:
        RuntimeError: If an unexpected error occurs while analysing mode and sentiment.
    """

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
            "text": (
                "Analyse the given video transcription and return:"
                "What is the overall mode of the video?"
                "What is the sentiment of the video?"
                "Briefly explain the reasons for choosing these labels."
                f"Transcription: {transcription}"
            )
        }
    ]

    # Define JSON schema
    json_schema = {
        "format": {
            "type": "json_schema",
            "name": "content_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "The overall mode or style of the video."
                    },
                    "sentiment": {
                        "type": "string",
                        "description": "The overall emotional tone of the video."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation for the mode and sentiment."
                    }
                },
                "required": ["mode", "sentiment", "explanation"],
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
        raise RuntimeError(f"Unexpected error occurred while analysing mode and sentiment: {e}") from e
    
    return response.output_text
