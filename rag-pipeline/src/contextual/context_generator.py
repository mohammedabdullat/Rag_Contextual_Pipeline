"""
Contextual Retrieval — Anthropic's technique implemented with any OpenAI-compatible LLM.

Reference: https://www.anthropic.com/news/contextual-retrieval

For each chunk, we ask the LLM:
  "Given the full document, write a short context that situates this chunk
   within the document."

The generated context is prepended to the chunk text before embedding/indexing,
dramatically improving retrieval quality.
"""
from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI
from tqdm import tqdm
from groq import Groq


from src.config import get_settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

_CONTEXT_PROMPT = """\
Here is a document:
<document>
{document}
</document>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

Please give a short, succinct context (2-3 sentences) to situate this chunk \
within the overall document. Focus on what section it belongs to, what topic it \
covers, and how it relates to the document's main argument. \
Reply with ONLY the context text, no preamble."""


class ContextGenerator:
    """
    Prepends LLM-generated situating context to each chunk.

    This is the core of Anthropic's Contextual Retrieval approach:
    instead of embedding a bare chunk, we embed:
        "<situating context>\n\n<chunk text>"

    This gives the embedding model enough context to place the chunk
    correctly in semantic space.
    """

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        settings = get_settings()
        self._client = client or OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self._model = settings.llm_chat_model
        self._context_window_chars = settings.context_window_chars

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def enrich_chunks(
        self,
        chunks: list[Chunk],
        full_document: str,
        show_progress: bool = True,
    ) -> list[Chunk]:
        """
        Mutates each chunk's `contextualized_text` field in-place.

        Args:
            chunks:         List of Chunk objects to enrich.
            full_document:  The complete document text (used as context window).
            show_progress:  Show tqdm progress bar.

        Returns:
            The same list of chunks (mutated).
        """
        # Truncate document to avoid massive token counts
        doc_excerpt = full_document[: self._context_window_chars]

        iterator = tqdm(chunks, desc="Generating chunk contexts") if show_progress else chunks
        success_count = 0

        for chunk in iterator:
            context = self._generate_context(doc_excerpt, chunk.text)
            if context:
                chunk.contextualized_text = f"{context}\n\n{chunk.text}"
                success_count += 1
            else:
                chunk.contextualized_text = chunk.text  # fallback

        logger.info(
            "Enriched %d/%d chunks with contextual context",
            success_count,
            len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _generate_context(self, document: str, chunk_text: str) -> Optional[str]:
        """Call the LLM to generate a situating context for a single chunk."""
        prompt = _CONTEXT_PROMPT.format(document=document, chunk=chunk_text)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10000,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Context generation failed for chunk: %s", exc)
            return None
