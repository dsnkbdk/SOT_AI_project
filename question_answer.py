import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

def question_answer(client: OpenAI, transcription: str, model: str) -> str:
    """
    Convert a video transcription into a list of question-answer (Q&A) pairs about the video.
    
    Args:
        client (OpenAI): An initialised OpenAI client with a valid API key.
        transcription (str): The complete transcription of the video.
        model (str): Model ID used to generate the response, like gpt-4o or o3.
    
    Returns:
        str: A JSON-formatted string containing a list of Q&A pairs.
    
    Raises:
        RuntimeError: If an unexpected error occurs while generating Q&A pairs.
    """

    # Build input content
    dev_content = [
        {
            "type": "input_text",
            "text": (
                "Each Q&A pair should be relevant to the video transcription content and provide a concise answer."
                "Return results that strictly match the given JSON format."
            )
        }
    ]

    usr_content = [
        {
            "type": "input_text",
            "text": (
                "Based on the given video transcription, generate a list of 5 to 10 useful Question-Answer (Q&A) pairs."
                f"Transcription: {transcription}"
            )
        }
    ]

    # Define JSON schema
    json_schema = {
        "format": {
            "type": "json_schema",
            "name": "Question_Answer",
            "schema": {
                "type": "object",
                "properties": {
                    "QA_pairs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Q": {"type": "string"},
                                "A": {"type": "string"}
                            },
                            "required": ["Q", "A"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["QA_pairs"],
                "additionalProperties": False
            }
        }
    }

    # Call OpenAI API
    try:
        logger.info("Generating Q&A pairs...")
        response = client.responses.create(
            model=model,
            input=[
                {"role": "developer", "content": dev_content},
                {"role": "user", "content": usr_content}
            ],
            text=json_schema
        )
    
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred while generating Q&A pairs") from e
    
    return response.output_text
