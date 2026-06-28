import re
import math
from typing import List, Any
from core.adapters.base import SalesEmail, Chunk
from core.chunking.base import BaseChunker

class SemanticChunker(BaseChunker):
    """
    Chunks email bodies using semantic similarity of sentences.
    Falls back to paragraph splitting if no embedder is configured.
    """
    def __init__(self, embedder: Any = None, threshold_percentile: float = 0.7, prepend_subject: bool = True):
        self.embedder = embedder
        self.threshold_percentile = threshold_percentile
        self.prepend_subject = prepend_subject

    def chunk(self, email: SalesEmail) -> List[Chunk]:
        body = email.body
        if not body:
            return []
            
        # If embedder is not provided, fall back to paragraph/sentence grouping
        if not self.embedder:
            return self._fallback_chunk(email)
            
        # 1. Split body into individual sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', body)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return self._create_chunk(email, sentences[0], 0)

        # 2. Get embeddings for all sentences
        try:
            embeddings = self.embedder.embed_documents(sentences)
        except Exception:
            # If embedding fails, fallback
            return self._fallback_chunk(email)

        # 3. Calculate cosine similarities between consecutive sentences
        distances = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            distances.append(1.0 - sim) # cosine distance

        if not distances:
            return self._create_chunk(email, " ".join(sentences), 0)

        # 4. Determine splitting threshold (using percentile of distances)
        sorted_distances = sorted(distances)
        threshold_idx = int(len(sorted_distances) * self.threshold_percentile)
        threshold = sorted_distances[min(threshold_idx, len(sorted_distances) - 1)]

        # 5. Split sentences into groups where distance > threshold
        chunks_content = []
        current_group = [sentences[0]]
        
        for i in range(len(sentences) - 1):
            if distances[i] > threshold:
                # Split here
                chunks_content.append(" ".join(current_group))
                current_group = [sentences[i+1]]
            else:
                current_group.append(sentences[i+1])
        if current_group:
            chunks_content.append(" ".join(current_group))

        # 6. Build chunk objects
        chunks = []
        for idx, text_group in enumerate(chunks_content):
            chunks.append(self._create_chunk(email, text_group, idx))
            
        return chunks

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm_v1 = math.sqrt(sum(x * x for x in v1))
        norm_v2 = math.sqrt(sum(x * x for x in v2))
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def _fallback_chunk(self, email: SalesEmail) -> List[Chunk]:
        # Simple fallback to splitting by paragraph
        paragraphs = [p.strip() for p in email.body.split("\n\n") if p.strip()]
        if not paragraphs:
            return []
        
        chunks = []
        for idx, para in enumerate(paragraphs):
            chunks.append(self._create_chunk(email, para, idx))
        return chunks

    def _create_chunk(self, email: SalesEmail, text: str, idx: int) -> Chunk:
        chunk_content = text
        if self.prepend_subject and email.subject:
            chunk_content = f"Subject: {email.subject}\nContent: {text}"
            
        return Chunk(
            chunk_id=f"{email.email_id}_sem_{idx}",
            content=chunk_content,
            email_id=email.email_id,
            sequence_id=email.sequence_id,
            step=email.step,
            persona=email.persona,
            industry=email.industry,
            stage=email.stage,
            open_rate=email.open_rate,
            reply_rate=email.reply_rate,
            source_metadata={"semantic_index": idx, **email.metadata}
        )
