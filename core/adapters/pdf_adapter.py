import fitz  # PyMuPDF
import re
import logging
from typing import List, Optional
from core.adapters.base import BaseAdapter, SalesEmail

logger = logging.getLogger(__name__)


def _extract_field(segment: str, label_pattern: str, multiword: bool = False) -> Optional[str]:
    """
    Extract a field value from a PDF text segment.
    
    Handles TWO common PDF text layouts:
      1. Inline colon format:   "Open Rate: 0.58"
      2. Newline-separated:     "Open Rate\n0.58"  (PyMuPDF default)
    """
    # 1. Try colon-delimited on same line:  "Label: value"
    colon_pat = rf'(?i)\b{label_pattern}\b\s*:\s*([^\n]+)'
    m = re.search(colon_pat, segment)
    if m:
        return m.group(1).strip()
    
    # 2. Try newline-separated:  "Label\nvalue"
    # Use MULTILINE so ^ anchors work per-line
    if multiword:
        # For multi-word labels like "Open Rate", match the whole phrase then grab next line
        newline_pat = rf'(?im)^{label_pattern}\s*$\s*\n\s*(.+)'
    else:
        newline_pat = rf'(?im)^{label_pattern}\s*$\s*\n\s*(.+)'
    m = re.search(newline_pat, segment)
    if m:
        return m.group(1).strip()
    
    return None


class PDFAdapter(BaseAdapter):
    """
    Adapter to parse and normalize sales sequence PDF files.
    Extracts text page-by-page and parses email segments.
    
    Handles PDFs where structured fields (Sequence, Step, Persona, etc.)
    appear as either colon-separated or newline-separated key-value pairs.
    """
    def adapt(self, source_path: str) -> List[SalesEmail]:
        logger.info(f"Loading and normalizing PDF data from {source_path}")
        emails = []
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(source_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n--- PAGE_BREAK ---\n"
            
            logger.debug(f"Extracted PDF text length: {len(full_text)} chars")
            
            # Split using Subject: or Subject\n headers
            # Handle both "Subject: ..." and "Subject\n..." formats
            segments = re.split(r'(?i)(?:^|\n)\s*Subject\s*[:\n]', full_text)
            
            if len(segments) <= 1:
                # Fallback: try splitting on PAGE_BREAK
                segments = [s.strip() for s in full_text.split("--- PAGE_BREAK ---") if s.strip()]
                if not segments:
                    logger.warning("No parseable email segments found in PDF.")
                    return []
                # For page-based segments, try to find Subject line within each
                new_segments = []
                for seg in segments:
                    subj_match = re.search(r'(?i)Subject\s*[:\n]\s*(.+)', seg)
                    if subj_match:
                        # Re-split at that Subject marker
                        idx = subj_match.start()
                        new_segments.append(seg[idx:])
                    else:
                        new_segments.append(seg)
                segments = new_segments
                
            parsed_segments = segments[1:] if not segments[0].strip() else segments
            # If first segment is clearly not an email (no body text), skip it
            if parsed_segments and len(parsed_segments[0].strip()) < 10:
                parsed_segments = parsed_segments[1:] if len(parsed_segments) > 1 else parsed_segments
            
            for idx, segment in enumerate(parsed_segments, start=1):
                lines = [l.strip() for l in segment.split('\n') if l.strip()]
                if not lines:
                    continue
                
                # Subject is the first line of the segment
                subject = lines[0].strip()
                # Clean up any residual "Subject:" prefix
                subject = re.sub(r'(?i)^Subject\s*:\s*', '', subject).strip()
                
                # --- Extract structured fields ---
                
                # Step (numeric)
                step_val = _extract_field(segment, r'Step')
                if step_val:
                    # Extract just the number
                    num_match = re.search(r'(\d+)', step_val)
                    step = int(num_match.group(1)) if num_match else idx
                else:
                    step = idx
                
                # Persona
                persona = _extract_field(segment, r'Persona') or "Unknown"
                
                # Industry
                industry = _extract_field(segment, r'Industry') or "General"
                
                # Stage
                stage = _extract_field(segment, r'Stage') or "awareness"
                
                # Open Rate
                open_rate_str = _extract_field(segment, r'Open\s*Rate', multiword=True)
                if open_rate_str:
                    rate_match = re.search(r'([\d\.]+)', open_rate_str)
                    open_rate = float(rate_match.group(1)) if rate_match else 0.0
                else:
                    open_rate = 0.0
                
                # Reply Rate
                reply_rate_str = _extract_field(segment, r'Reply\s*Rate', multiword=True)
                if reply_rate_str:
                    rate_match = re.search(r'([\d\.]+)', reply_rate_str)
                    reply_rate = float(rate_match.group(1)) if rate_match else 0.0
                else:
                    reply_rate = 0.0
                
                # Sequence name
                seq_raw = _extract_field(segment, r'Sequence(?:\s*ID)?')
                if seq_raw:
                    # Clean: remove any lines that look like metadata labels
                    seq_raw = re.split(r'(?i)\b(?:Step|Persona|Industry|Stage|Open|Reply)\b', seq_raw)[0].strip()
                    sequence_id = re.sub(r'[^a-zA-Z0-9]+', '_', seq_raw).strip('_').lower() or "seq_pdf"
                else:
                    sequence_id = "seq_pdf"
                
                # --- Extract email body ---
                body_match = re.search(r'(?i)\b(?:Body|Content|Email Content)\b\s*[:\n]\s*(.*)', segment, re.DOTALL)
                if body_match:
                    body = body_match.group(1).strip()
                    # Trim trailing page breaks and metadata
                    body = re.split(r'---\s*PAGE_BREAK\s*---', body)[0].strip()
                else:
                    # Fallback: grab everything after the last known field
                    body = "\n".join(lines[1:]).strip()
                
                email = SalesEmail(
                    email_id=f"email_pdf_{idx}",
                    sequence_id=sequence_id,
                    step=step,
                    subject=subject,
                    body=body,
                    persona=persona,
                    industry=industry,
                    stage=stage,
                    open_rate=open_rate,
                    reply_rate=reply_rate,
                    metadata={"source": "pdf", "source_file": source_path}
                )
                emails.append(email)
                logger.info(
                    f"  Parsed email #{idx}: seq='{sequence_id}' step={step} "
                    f"persona='{persona}' open_rate={open_rate} reply_rate={reply_rate} "
                    f"subject='{subject[:50]}...'"
                )
                
            # If nothing was parsed, fallback to raw page-based chunks
            if not emails:
                logger.warning("Primary parsing yielded no emails. Falling back to page-based extraction.")
                pages = full_text.split("--- PAGE_BREAK ---")
                for p_idx, page in enumerate(pages):
                    page_clean = page.strip()
                    if not page_clean:
                        continue
                    lines = [l.strip() for l in page_clean.split('\n') if l.strip()]
                    if not lines:
                        continue
                    subject = lines[0]
                    body = "\n".join(lines[1:])
                    
                    email = SalesEmail(
                        email_id=f"email_pdf_{p_idx}",
                        sequence_id="seq_pdf_fallback",
                        step=p_idx + 1,
                        subject=subject,
                        body=body,
                        persona="Unknown",
                        industry="General",
                        stage="awareness",
                        open_rate=0.0,
                        reply_rate=0.0,
                        metadata={"source": "pdf_fallback"}
                    )
                    emails.append(email)
                    
        except Exception as e:
            logger.error(f"Failed to read PDF: {e}")
            raise
        
        logger.info(f"PDF adapter extracted {len(emails)} email(s) total.")
        return emails
