import csv
import logging
from typing import List, Dict, Any
from core.adapters.base import BaseAdapter, SalesEmail

logger = logging.getLogger(__name__)

class CSVAdapter(BaseAdapter):
    """
    Adapter to parse and normalize sales sequence CSV files.
    """
    def __init__(self, column_mapping: Dict[str, str] = None):
        # Default mapping from CSV headers to standard schema attributes
        self.column_mapping = column_mapping or {
            "email_id": "email_id",
            "sequence_id": "sequence_id",
            "step": "step",
            "subject": "subject",
            "body": "body",
            "persona": "persona",
            "industry": "industry",
            "stage": "stage",
            "open_rate": "open_rate",
            "reply_rate": "reply_rate"
        }

    def adapt(self, source_path: str) -> List[SalesEmail]:
        logger.info(f"Loading and normalizing CSV data from {source_path}")
        emails = []
        
        try:
            with open(source_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    logger.warning(f"CSV file {source_path} is empty or has no headers.")
                    return []
                
                # Check mapping against actual headers (case insensitive / fuzzy match fallbacks)
                headers = [h.strip() for h in reader.fieldnames]
                resolved_mapping = self._resolve_headers(headers)
                
                for row_idx, row in enumerate(reader):
                    try:
                        email_id = row.get(resolved_mapping.get("email_id", ""), f"email_{row_idx}")
                        sequence_id = row.get(resolved_mapping.get("sequence_id", ""), "seq_default")
                        
                        # Step conversion
                        step_val = row.get(resolved_mapping.get("step", ""), "1")
                        step = int(float(step_val)) if step_val else 1
                        
                        subject = row.get(resolved_mapping.get("subject", ""), "").strip()
                        body = row.get(resolved_mapping.get("body", ""), "").strip()
                        persona = row.get(resolved_mapping.get("persona", ""), "Unknown").strip()
                        industry = row.get(resolved_mapping.get("industry", ""), "General").strip()
                        stage = row.get(resolved_mapping.get("stage", ""), "awareness").strip()
                        
                        # Rate conversions
                        open_rate_val = row.get(resolved_mapping.get("open_rate", ""), "0.0")
                        open_rate = float(open_rate_val) if open_rate_val else 0.0
                        
                        reply_rate_val = row.get(resolved_mapping.get("reply_rate", ""), "0.0")
                        reply_rate = float(reply_rate_val) if reply_rate_val else 0.0
                        
                        # Additional fields go to metadata
                        metadata = {}
                        for k, v in row.items():
                            if k not in resolved_mapping.values():
                                metadata[k] = v
                                
                        email = SalesEmail(
                            email_id=email_id,
                            sequence_id=sequence_id,
                            step=step,
                            subject=subject,
                            body=body,
                            persona=persona,
                            industry=industry,
                            stage=stage,
                            open_rate=open_rate,
                            reply_rate=reply_rate,
                            metadata=metadata
                        )
                        emails.append(email)
                    except Exception as row_err:
                        logger.error(f"Error parsing row {row_idx}: {row_err}. Skipping row.")
                        
        except FileNotFoundError:
            logger.error(f"CSV file not found: {source_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise
            
        return emails

    def _resolve_headers(self, headers: List[str]) -> Dict[str, str]:
        """
        Fuzzy matches actual CSV headers to desired keys.
        """
        resolved = {}
        for target_key, default_name in self.column_mapping.items():
            # Try exact match
            if default_name in headers:
                resolved[target_key] = default_name
                continue
            # Try case-insensitive matching
            matched = False
            for header in headers:
                if header.lower() == default_name.lower():
                    resolved[target_key] = header
                    matched = True
                    break
            if matched:
                continue
                
            # Fallbacks
            fallbacks = {
                "email_id": ["id", "email id"],
                "sequence_id": ["seq_id", "sequence id", "sequence"],
                "body": ["content", "text", "email_body", "email body"],
                "step": ["sequence_step", "email_step", "order"],
                "subject": ["email_subject", "email subject"],
            }
            if target_key in fallbacks:
                for fb in fallbacks[target_key]:
                    for header in headers:
                        if header.lower() == fb:
                            resolved[target_key] = header
                            matched = True
                            break
                    if matched:
                        break
            
            # Default to target key if not found
            if not matched:
                resolved[target_key] = target_key
                
        return resolved
