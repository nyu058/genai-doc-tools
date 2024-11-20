import os
from dotenv import load_dotenv

load_dotenv()

class DocumentAiTool:

    prompt_template = ""
    model = ""

    def __init__(self) -> None:
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        