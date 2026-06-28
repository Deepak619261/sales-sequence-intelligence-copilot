import re
from typing import List
from core.adapters.base import SalesEmail, Chunk
from core.chunking.base import BaseChunker

class SentenceChunker(BaseChunker):
    """
    Chunks email bodies into groups of sentences.
    """
    def __init__(self, sentences_per_chunk: int = 2, prepend_subject: bool = True):
        self.sentences_per_chunk = sentences_per_chunk
        self.prepend_subject = prepend_subject

    def chunk(self, email: SalesEmail) -> List[Chunk]:
        chunks = []
        body = email.body
        if not body:
            return []
            
        # Regex to split body into sentences (handles '.', '!', '?' followed by spaces or newlines)
        raw_sentences = re.split(r'(?<=[.!?])\s+', body)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        if not sentences:
            return []
            
        # Group sentences based on sentences_per_chunk setting
        grouped_sentences = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            group = " ".join(sentences[i:i + self.sentences_per_chunk])
            grouped_sentences.append(group)
            
        for idx, text_group in enumerate(grouped_sentences):
            chunk_content = text_group
            if self.prepend_subject and email.subject:
                chunk_content = f"Subject: {email.subject}\nContent: {text_group}"
                
            chunk_id = f"{email.email_id}_sent_{idx}"
            
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
                    "original_chunk_index": idx,
                    "sentences_count": len(text_group.split('. ')),
                    **email.metadata
                }
            )
            chunks.append(chunk)
            
        return chunks
