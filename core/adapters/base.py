import abc
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SalesEmail:
    email_id: str
    sequence_id: str
    step: int
    subject: str
    body: str
    persona: str
    industry: str
    stage: str
    open_rate: float
    reply_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email_id": self.email_id,
            "sequence_id": self.sequence_id,
            "step": self.step,
            "subject": self.subject,
            "body": self.body,
            "persona": self.persona,
            "industry": self.industry,
            "stage": self.stage,
            "open_rate": self.open_rate,
            "reply_rate": self.reply_rate,
            "metadata": self.metadata
        }

@dataclass
class SalesSequence:
    sequence_id: str
    name: str
    emails: List[SalesEmail] = field(default_factory=list)

@dataclass
class Chunk:
    chunk_id: str
    content: str
    email_id: str
    sequence_id: str
    step: int
    persona: str
    industry: str
    stage: str
    open_rate: float
    reply_rate: float
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "email_id": self.email_id,
            "sequence_id": self.sequence_id,
            "step": self.step,
            "persona": self.persona,
            "industry": self.industry,
            "stage": self.stage,
            "open_rate": self.open_rate,
            "reply_rate": self.reply_rate,
            "source_metadata": self.source_metadata
        }

class BaseAdapter(abc.ABC):
    """
    Interface for normalization adapters. Reads from a raw source 
    and transforms data into the standard SalesSequence schema.
    """
    @abc.abstractmethod
    def adapt(self, source_path: str) -> List[SalesEmail]:
        pass
