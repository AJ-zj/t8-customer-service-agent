"""Deterministic, bounded retrieval query from the current turn and recent context."""


MAX_TOOL_QUESTION_LENGTH = 500


def build_contextual_retrieval_question(question: str, conversation_context=None) -> str:
    """Prefix only the latest prior user question; never use model tool arguments."""
    for message in reversed(conversation_context or []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            previous_question = message["content"].strip()
            if previous_question:
                combined = f"{previous_question} {question}"
                return combined if len(combined) <= MAX_TOOL_QUESTION_LENGTH else question
    return question
