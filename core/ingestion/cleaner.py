import re
import unicodedata
from typing import Dict, Any
from core.adapters.base import SalesEmail

class DataCleaner:
    """
    Cleans raw text fields in SalesEmail records.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        
        # 1. Normalize Unicode (e.g., curly quotes, accents)
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Strip HTML tags if any are present
        text = re.sub(r"<[^>]+>", " ", text)
        
        # 3. Standardize whitespace (multiple spaces/tabs to a single space)
        # Note: we want to keep line breaks if they format the email structure, but avoid excess empty lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        
        return text.strip()

    def clean(self, email: SalesEmail) -> SalesEmail:
        """
        Returns a new SalesEmail instance with cleaned text properties.
        """
        cleaned_subject = self.clean_text(email.subject)
        cleaned_body = self.clean_text(email.body)
        
        # Safe rate clamp between 0.0 and 1.0
        open_rate = max(0.0, min(1.0, email.open_rate))
        reply_rate = max(0.0, min(1.0, email.reply_rate))
        
        return SalesEmail(
            email_id=email.email_id,
            sequence_id=email.sequence_id,
            step=email.step,
            subject=cleaned_subject,
            body=cleaned_body,
            persona=email.persona,
            industry=email.industry,
            stage=email.stage,
            open_rate=open_rate,
            reply_rate=reply_rate,
            metadata=email.metadata
        )
