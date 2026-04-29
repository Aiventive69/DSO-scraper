"""
Kadaster BAG API client (Individuele Bevragingen v2).
Provides robust object-identification context for an address.
"""

import logging
import re
from typing import Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

BAG_BASE = "https://api.bag.kadaster.nl/lvbag/individuelebevragingen/v2"


def _parse_postcode_and_number(adres_display: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Parse 'Straat 12A-1, 1234AB Plaats' into
    (postcode, huisnummer, huisletter, huisnummertoevoeging).
    """
    if not adres_display:
        return None, None, None, None

    postcode_match = re.search(r"\b(\d{4}\s?[A-Z]{2})\b", adres_display.upper())
    postcode = postcode_match.group(1).replace(" ", "") if postcode_match else None

    # Search number+optional suffix before comma
    first_part = adres_display.split(",")[0]
    num_match = re.search(r"\b(\d+)\s*([A-Za-z]?)\s*(?:[-/]?\s*([A-Za-z0-9]+))?\b", first_part)
    if not num_match:
        return postcode, None, None, None

    huisnummer = int(num_match.group(1))
    huisletter = (num_match.group(2) or "").upper() or None
    toevoeging = (num_match.group(3) or "").upper() or None
    return postcode, huisnummer, huisletter, toevoeging


async def fetch_bag_context(adres_display: str, bag_api_key: str) -> Dict:
    """
    Fetch BAG 'adres uitgebreid' context by postcode/huisnummer.
    Returns normalized dict with object identifiers and key building data.
    """
    if not bag_api_key:
        return {}

    postcode, huisnummer, huisletter, toevoeging = _parse_postcode_and_number(adres_display)
    if not postcode or not huisnummer:
        logger.info("BAG skip: kon geen postcode/huisnummer parsen uit adres")
        return {}

    params = {
        "postcode": postcode,
        "huisnummer": str(huisnummer),
        "exacteMatch": "true",
        "page": "1",
        "pageSize": "5",
    }
    if huisletter:
        params["huisletter"] = huisletter
    if toevoeging:
        params["huisnummertoevoeging"] = toevoeging

    headers = {
        "X-Api-Key": bag_api_key,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # Use "adressen" endpoint first (stable), then enrich if needed.
            resp = await client.get(f"{BAG_BASE}/adressen", params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as e:
        logger.warning(f"BAG query failed: {e}")
        return {}

    items = payload.get("_embedded", {}).get("adressen", []) or payload.get("_embedded", {}).get("adressenUitgebreid", [])
    if not items:
        return {}

    best = items[0]
    return {
        "openbare_ruimte": best.get("openbareRuimteNaam", ""),
        "huisnummer": best.get("huisnummer"),
        "huisletter": best.get("huisletter"),
        "huisnummertoevoeging": best.get("huisnummertoevoeging"),
        "postcode": best.get("postcode"),
        "woonplaats": best.get("woonplaatsNaam", ""),
        "nummeraanduiding_id": best.get("nummeraanduidingIdentificatie", ""),
        "adresseerbaar_object_id": best.get("adresseerbaarObjectIdentificatie", ""),
        "pand_ids": best.get("pandIdentificaties", []) or [],
        "adresregel5": best.get("adresregel5", ""),
        "adresregel6": best.get("adresregel6", ""),
    }


def format_bag_for_llm(bag_data: Dict) -> str:
    if not bag_data:
        return ""
    lines = [
        "## BAG context (Kadaster)",
        f"Adres: {bag_data.get('adresregel5', '')}",
        f"Plaats: {bag_data.get('adresregel6', '')}",
        f"Nummeraanduiding ID: {bag_data.get('nummeraanduiding_id', '')}",
        f"Adresseerbaar object ID: {bag_data.get('adresseerbaar_object_id', '')}",
    ]
    pand_ids = bag_data.get("pand_ids", [])
    if pand_ids:
        lines.append(f"Pand ID(s): {', '.join(pand_ids)}")
    return "\n".join(lines)
