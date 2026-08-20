"""
load_documents.py

Extracts text from the 5 company annual report PDFs.
Each PDF's text is pulled page by page, so we know which page
each piece of text came from. This page number gets carried
forward through chunking, so citations later can point to a
real page in the original report.
"""

import pdfplumber

# Hardcoded company info. Filenames must match what's inside data/.
# sector is stored here too since the dropdown and future features
# (like future sector-based grouping) will need it.
COMPANIES = {
    "HDFC Bank": {"file": "data/HDFC.pdf", "sector": "Banking and Finance"},
    "TCS": {"file": "data/TCS.pdf", "sector": "IT and Technology"},
    "HUL": {"file": "data/HUL.pdf", "sector": "FMCG"},
    "Reliance Industries": {"file": "data/RIL.pdf", "sector": "Energy and Oil and Gas"},
    "Sun Pharma": {"file": "data/SunPharma.pdf", "sector": "Pharma and Healthcare"},
}


def load_pdf_text(filepath):
    """
    Opens one PDF and extracts text page by page.
    Returns a list of dicts, one per page, like:
    [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    Pages with no extractable text (blank pages, pure image pages)
    are skipped, not included as empty entries. This is logged so
    we know if a report has more blank/image pages than expected.
    """
    pages_data = []
    skipped_pages = 0

    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({"page": i + 1, "text": text.strip()})
            else:
                skipped_pages += 1

    if skipped_pages > 0:
        print(f"  Note: {skipped_pages} page(s) had no extractable text, skipped.")

    return pages_data


def load_all_companies():
    """
    Loads text for every company in COMPANIES.
    Returns a dict like:
    {
        "HDFC Bank": {
            "sector": "Banking and Finance",
            "pages": [{"page": 1, "text": "..."}, ...]
        },
        ...
    }
    """
    all_data = {}

    for company_name, info in COMPANIES.items():
        print(f"Loading {company_name} ({info['file']})...")
        pages = load_pdf_text(info["file"])
        all_data[company_name] = {
            "sector": info["sector"],
            "pages": pages,
        }
        print(f"  Done. {len(pages)} pages with text extracted.")

    return all_data


# Quick manual test, run this file directly to check extraction works
# before moving on to chunking.
if __name__ == "__main__":
    data = load_all_companies()

    print("\nSummary:")
    for company, info in data.items():
        print(f"  {company}: {len(info['pages'])} pages, sector: {info['sector']}")
