"""
Answer Generator — calls OpenAI (or any compatible LLM) to produce answers
from retrieved context chunks.
"""
from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise research assistant. Answer questions using ONLY the provided \
context passages. If the context does not contain enough information to answer \
the question, say so clearly. Be concise and cite relevant passage numbers."""

_USER_TEMPLATE = """\
Context passages:
{context}

Question: {question}

Answer:"""


class AnswerGenerator:
    """
    Generates natural language answers from retrieved context using an LLM.

    Designed to be LLM-agnostic — just swap OPENAI_BASE_URL and
    OPENAI_CHAT_MODEL in .env to use a different provider.
    """

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        settings = get_settings()
        self._client = client or OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self._model = settings.llm_chat_model

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer for the given question using the context passages.

        Args:
            question: The user's question.
            context:  Retrieved context text (formatted by Retriever).

        Returns:
            Generated answer string.
        """
        user_msg = _USER_TEMPLATE.format(context=context, question=question)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=7000,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            return f"Error generating answer: {exc}"
