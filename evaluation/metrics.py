from typing import List, Set

class RetrievalMetrics:
    """
    Computes standard retrieval evaluation metrics.
    """
    @staticmethod
    def precision_at_k(retrieved: List[str], ground_truth: List[str], k: int) -> float:
        """
        Precision@K = (Relevant documents retrieved) / K
        """
        if k <= 0:
            return 0.0
            
        retrieved_k = retrieved[:k]
        gt_set = set(ground_truth)
        
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in gt_set)
        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved: List[str], ground_truth: List[str], k: int) -> float:
        """
        Recall@K = (Relevant documents retrieved) / (Total relevant documents)
        """
        if not ground_truth:
            return 0.0
            
        retrieved_k = retrieved[:k]
        gt_set = set(ground_truth)
        
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in gt_set)
        return relevant_retrieved / len(ground_truth)

    @staticmethod
    def reciprocal_rank(retrieved: List[str], ground_truth: List[str], k: int = None) -> float:
        """
        Reciprocal Rank = 1 / (Rank of first relevant document)
        Returns 0.0 if no relevant document was retrieved up to K.
        """
        retrieved_slice = retrieved[:k] if k is not None else retrieved
        gt_set = set(ground_truth)
        for idx, doc_id in enumerate(retrieved_slice):
            if doc_id in gt_set:
                return 1.0 / (idx + 1)
        return 0.0

    @staticmethod
    def hit_rate(retrieved: List[str], ground_truth: List[str], k: int) -> float:
        """
        Hit Rate = 1 if at least one relevant document is in the top K, else 0.
        """
        retrieved_k = retrieved[:k]
        gt_set = set(ground_truth)
        
        for doc_id in retrieved_k:
            if doc_id in gt_set:
                return 1.0
        return 0.0
