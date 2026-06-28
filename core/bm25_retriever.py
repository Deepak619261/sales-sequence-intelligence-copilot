import re
import math
import logging
from typing import List, Dict, Any, Tuple, Set
from core.adapters.base import Chunk
from core.vectorstore.base import BaseVectorStore
from core.base import BaseRetriever

logger = logging.getLogger(__name__)

class BM25Retriever(BaseRetriever):
    """
    Pure-Python Okapi BM25 retriever for exact keyword search.
    Pre-filters candidate chunks based on metadata constraints.
    """
    def __init__(
        self, 
        vector_store: BaseVectorStore,
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.vector_store = vector_store
        self.k1 = k1
        self.b = b
        
        # English stop words list to remove noise from term counts
        self.stop_words: Set[str] = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent",
            "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant",
            "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he",
            "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows",
            "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "itself", "lets", "me",
            "more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
            "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell",
            "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs",
            "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve",
            "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed",
            "well", "were", "weve", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while",
            "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre",
            "youve", "your", "yours", "yourself", "yourselves"
        }

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, find all word characters, filter out stopwords
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in self.stop_words]

    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Chunk, float]]:
        logger.info(f"Retrieving top {k} BM25 keyword matches for query: '{query}'")
        
        # 1. Fetch all documents from the vector store
        try:
            all_chunks = self.vector_store.get_all_chunks()
        except Exception as e:
            logger.error(f"Failed to fetch documents for BM25: {e}")
            return []

        # 2. Apply metadata filter (Logical AND)
        filtered_chunks: List[Chunk] = []
        filter_dict = filter_dict or {}
        for chunk in all_chunks:
            match = True
            for f_key, f_val in filter_dict.items():
                if getattr(chunk, f_key, None) != f_val:
                    match = False
                    break
            if match:
                filtered_chunks.append(chunk)

        N = len(filtered_chunks)
        if N == 0:
            logger.info("No documents matched the metadata filter. BM25 returned 0 hits.")
            return []

        # 3. Tokenize query and docs
        query_terms = self._tokenize(query)
        if not query_terms:
            logger.warning("Empty search query after stopword tokenization.")
            return [(chunk, 0.0) for chunk in filtered_chunks[:k]]

        # Tokenize documents
        doc_tokens = [self._tokenize(chunk.content) for chunk in filtered_chunks]
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        avg_doc_len = sum(doc_lengths) / N if N > 0 else 0

        # 4. Calculate Document Frequencies (df) for each term in the query
        df: Dict[str, int] = {}
        for term in query_terms:
            df[term] = sum(1 for tokens in doc_tokens if term in tokens)

        # 5. Compute Okapi BM25 scores
        scored_chunks = []
        for idx, chunk in enumerate(filtered_chunks):
            score = 0.0
            tokens = doc_tokens[idx]
            doc_len = doc_lengths[idx]
            
            # Count word frequencies in this document
            tf: Dict[str, int] = {}
            for term in tokens:
                tf[term] = tf.get(term, 0) + 1

            for term in query_terms:
                if term not in tf:
                    continue
                    
                # Term's document frequency
                term_df = df[term]
                
                # Inverse Document Frequency (IDF)
                # Max formula avoids negative idfs for very frequent words
                idf = math.log((N - term_df + 0.5) / (term_df + 0.5) + 1.0)
                idf = max(0.0001, idf) # Floor at a tiny positive number
                
                term_tf = tf[term]
                
                # BM25 tf scaling
                numerator = term_tf * (self.k1 + 1.0)
                denominator = term_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len if avg_doc_len > 0 else 1.0))
                
                score += idf * (numerator / denominator)
                
            scored_chunks.append((chunk, score))

        # Sort descending by BM25 score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"BM25 search completed. Found {len(scored_chunks)} candidates scored.")
        return scored_chunks[:k]
