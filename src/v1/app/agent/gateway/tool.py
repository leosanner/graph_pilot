import structlog
from langchain_core.tools import BaseTool, tool

REPLY_TO_USER = "reply_to_user"
DELEGATE_TO_RETRIEVAL = "delegate_to_retrieval"


def make_gateway_tools() -> list[BaseTool]:
    logger = structlog.get_logger()

    @tool(REPLY_TO_USER)
    def reply_to_user(response: str) -> str:
        """Reply when the user does not need the ingested documents.

        Use for greetings, thanks, small talk, and anything answerable
        without searching the local PDF collection. Pass the full reply
        in `response`. Do not call this for factual questions that may
        be in the user's documents.

        Args:
            response: The complete reply to show the user.
        """

        logger.info("tool.reply_to_user")
        return response

    @tool(DELEGATE_TO_RETRIEVAL)
    def delegate_to_retrieval() -> str:
        """Hand off to retrieval when the question may be in the ingested PDFs.

        Call this instead of answering those questions from memory. Do
        not use for greetings, thanks, or small talk.
        """

        logger.info("tool.delegate_to_retrieval")
        return "Continuing with document retrieval."

    return [reply_to_user, delegate_to_retrieval]
