"""
DSO vergunningcheck/toepasbare-regels connector.

Because endpoint paths vary per subscribed service package, this client:
- uses an optional env-configured endpoint if provided
- otherwise tries a small set of likely search paths
- never crashes the app; returns empty context on failure
"""

import logging
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TR_BASE = "https://service.omgevingswet.overheid.nl/publiek/toepasbare-regels/api/zoekinterface/v2"


async def fetch_vergunningcheck_context(
    vraag: str,
    dso_api_key: str,
    endpoint_override: Optional[str] = None,
) -> List[Dict]:
    """
    Try to fetch applicability/search hints for vergunningcheck-like activities.
    Returns list of normalized activity-like hits.
    """
    if not dso_api_key or not vraag.strip():
        return []

    base = (endpoint_override or DEFAULT_TR_BASE).rstrip("/")
    candidates = [
        (f"{base}/activiteiten/_zoek", "activiteiten"),
        (f"{base}/werkzaamheden/_zoek", "werkzaamheden"),
    ]
    headers = {
        "X-Api-Key": dso_api_key,
        "Accept": "application/hal+json",
        "Content-Type": "application/json",
    }
    payload = {"zoekTerm": vraag, "grootte": 10}

    async with httpx.AsyncClient(timeout=15) as client:
        for url, kind in candidates:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in (401, 403):
                    logger.warning(f"Vergunningcheck unauthorized for endpoint {url}")
                    return []
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                continue

            # Normalize several likely response shapes
            items = (
                data.get("_embedded", {}).get(kind)
                or data.get("_embedded", {}).get("werkzaamheden")
                or data.get("_embedded", {}).get("activiteiten")
                or data.get("werkzaamheden")
                or data.get("activiteiten")
                or data.get("results")
                or []
            )
            out: List[Dict] = []
            for it in items[:10]:
                out.append(
                    {
                        "naam": it.get("naam") or it.get("omschrijving") or it.get("title") or "Onbekend",
                        "type": it.get("type") or it.get("soort") or "",
                        "referentie": it.get("functioneleStructuurReferentie") or it.get("id") or "",
                    }
                )
            return out

    return []


def format_vergunningcheck_for_llm(items: List[Dict]) -> str:
    if not items:
        return ""
    lines = ["## Vergunningcheck context (DSO toepasbare regels zoeken)"]
    for it in items[:10]:
        lines.append(
            f"- {it.get('naam','Onbekend')} | type: {it.get('type','')} | ref: {it.get('referentie','')}"
        )
    return "\n".join(lines)
