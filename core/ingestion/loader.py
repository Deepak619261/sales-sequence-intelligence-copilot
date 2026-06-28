import logging
from typing import List
from core.adapters.base import BaseAdapter, SalesEmail
from core.ingestion.cleaner import DataCleaner

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Standardized loader that coordinates an adapter and a cleaner 
    to retrieve normalized, clean SalesEmails.
    """
    def __init__(self, adapter: BaseAdapter, cleaner: DataCleaner = None):
        self.adapter = adapter
        self.cleaner = cleaner or DataCleaner()

    def load(self, source_path: str) -> List[SalesEmail]:
        logger.info(f"Starting ingestion process for: {source_path}")
        try:
            # 1. Load data from target adapter
            raw_emails = self.adapter.adapt(source_path)
            logger.info(f"Adapter parsed {len(raw_emails)} records.")
            
            # 2. Clean data
            cleaned_emails = []
            for email in raw_emails:
                cleaned_emails.append(self.cleaner.clean(email))
                
            logger.info(f"Successfully cleaned and normalized {len(cleaned_emails)} records.")
            return cleaned_emails
            
        except Exception as e:
            logger.error(f"Ingestion loader failed: {e}")
            raise
