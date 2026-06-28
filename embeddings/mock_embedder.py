import re
import math
from typing import List
from embeddings.base import BaseEmbedder

class MockEmbedder(BaseEmbedder):
    """
    Deterministic concept-space embedder.
    Maps text to a normalized 10-dimensional thematic vector.
    This simulates neural semantic embedding distances without external packages.
    """
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        
        # Word lists defining each concept dimension
        self.concepts = {
            "scalability": ["scale", "scaling", "scalability", "throughput", "surge", "traffic", "growth", "high-throughput", "volume", "db"],
            "latency": ["latency", "spike", "spikes", "lag", "delay", "response", "speed", "slow", "fast", "milliseconds", "ms", "velocity"],
            "caching": ["cache", "caching", "redis", "buffer", "cache-hit", "edge", "memory", "cache-hits", "dashboards"],
            "compliance": ["compliance", "audit", "sec", "regulations", "rules", "legal", "regulatory", "audit-ready", "gap", "audits", "sec-compliant"],
            "logging": ["log", "logs", "logging", "audit-trail", "hash", "timestamp", "database-logs", "records", "database"],
            "recruiting": ["recruit", "recruiting", "hire", "hiring", "candidate", "candidates", "interview", "cv", "greenhouse", "ats", "resume", "resumes", "pre-screening", "screening", "hr", "recruitment"],
            "scheduling": ["schedule", "scheduling", "rotation", "shifts", "hours", "calendar", "double-booking", "calendars", "double-bookings", "shift", "bookings"],
            "healthcare": ["clinic", "clinical", "clinician", "physician", "healthcare", "doctor", "ops", "operations", "clinicians", "physicians"],
            "burnout": ["burnout", "overwork", "exhaustion", "stress", "volatility", "staff", "coordination", "burnouts"],
            "case_study": ["acmecorp", "acmebank", "bytescale", "stjude", "case study", "cases", "demo", "video", "acme", "link", "checklist", "checklist:"]
        }
        self.concept_keys = list(self.concepts.keys())

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _generate_embedding(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vector = [0.0] * 10
        
        # Calculate frequency weights for each concept category
        for idx, key in enumerate(self.concept_keys):
            word_list = self.concepts[key]
            match_count = sum(1 for token in tokens if token in word_list)
            vector[idx] = float(match_count)
            
        # Add a tiny default background noise to prevent zero-vectors
        for idx in range(len(vector)):
            vector[idx] += 0.05
            
        # Compute L2 Norm (length of the vector)
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
            
        # Pad/project the 10-dimensional concept vector to the target dimension (e.g. 768)
        # to ensure it matches the dimension parameter requested by factories.
        full_vector = [0.0] * self.dimension
        for i in range(self.dimension):
            # Deterministically map dimensions using modulo index
            full_vector[i] = vector[i % 10]
            
        # Re-normalize full projected vector
        fnorm = math.sqrt(sum(x*x for x in full_vector))
        if fnorm > 0:
            full_vector = [x / fnorm for x in full_vector]
            
        return full_vector

    def embed_query(self, text: str) -> List[float]:
        return self._generate_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_embedding(t) for t in texts]
