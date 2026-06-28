from typing import List
from core.adapters.base import SalesEmail, Chunk
from core.chunking.base import BaseChunker

class SlidingWindowChunker(BaseChunker):
    """
    Chunks email bodies using a sliding window approach with words.
    """
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20, prepend_subject: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prepend_subject = prepend_subject
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk(self, email: SalesEmail) -> List[Chunk]:
        chunks = []
        body = email.body
        if not body:
            return []
            
        words = body.split()
        if not words:
            return []
            
        total_words = len(words)
        
        # If the text is shorter than chunk_size, make a single chunk
        if total_words <= self.chunk_size:
            chunk_content = " ".join(words)
            if self.prepend_subject and email.subject:
                chunk_content = f"Subject: {email.subject}\nContent: {chunk_content}"
                
            return [Chunk(
                chunk_id=f"{email.email_id}_slide_0",
                content=chunk_content,
                email_id=email.email_id,
                sequence_id=email.sequence_id,
                step=email.step,
                persona=email.persona,
                industry=email.industry,
                stage=email.stage,
                open_rate=email.open_rate,
                reply_rate=email.reply_rate,
                source_metadata={"window_start": 0, "window_end": total_words, **email.metadata}
            )]
            
        idx = 0
        start = 0
        while start < total_words:
            end = start + self.chunk_size
            window_words = words[start:end]
            
            chunk_content = " ".join(window_words)
            if self.prepend_subject and email.subject:
                chunk_content = f"Subject: {email.subject}\nContent: {chunk_content}"
                
            chunk_id = f"{email.email_id}_slide_{idx}"
            
            chunk = Chunk(
                chunk_id=chunk_id,
                content=chunk_content,
                email_id=email.email_id,
                sequence_id=email.sequence_id,
                step=email.step,
                persona=email.persona,
                industry=email.industry,
                stage=email.stage,
                open_rate=email.open_rate,
                reply_rate=email.reply_rate,
                source_metadata={
                    "window_start": start,
                    "window_end": min(end, total_words),
                    **email.metadata
                }
            )
            chunks.append(chunk)
            
            # Slide window forward
            start += (self.chunk_size - self.chunk_overlap)
            idx += 1
            
        return chunks
