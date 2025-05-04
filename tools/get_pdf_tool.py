import json, base64
from openai import OpenAI
from agents.tool import FunctionTool

class GetResumePDFTool(FunctionTool):
    """
    Retrieve the original résumé PDF given its OpenAI file_id.
    Returns a data-URL so UIs can offer a download link.
    """
    def __init__(self):
        schema = {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "OpenAI file-ID of the résumé PDF"}
            },
            "required": ["file_id"],
            "additionalProperties": False
        }
        super().__init__(
            name="get_resume_pdf",
            description="Download the résumé PDF by file_id.",
            params_json_schema=schema,
            on_invoke_tool=self.on_invoke_tool
        )

    async def on_invoke_tool(self, _ctx, args: str) -> str:
        file_id   = json.loads(args)["file_id"]
        pdf_bytes = OpenAI().files.content(file_id)
        b64       = base64.b64encode(pdf_bytes).decode()
        return f"data:application/pdf;base64,{b64}"
