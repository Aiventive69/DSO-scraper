"""
PDOK Locatieserver geocoding service.
Converts Dutch addresses to RD New (EPSG:28992) coordinates.
"""

import re
import httpx
from typing import Optional, Tuple

PDOK_FREE_URL = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free"
PDOK_SUGGEST_URL = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/suggest"
PDOK_LOOKUP_URL = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/lookup"


def parse_rd_point(centroide_rd: str) -> Tuple[float, float]:
    """Parse 'POINT(139784 442870)' to (x, y) tuple in RD New."""
    match = re.match(r"POINT\(([0-9.]+)\s+([0-9.]+)\)", centroide_rd or "")
    if not match:
        raise ValueError(f"Cannot parse RD point: {centroide_rd}")
    return float(match.group(1)), float(match.group(2))


def parse_wgs84_point(centroide_ll: str) -> Tuple[float, float]:
    """Parse 'POINT(4.89 52.37)' to (lon, lat) tuple in WGS84."""
    match = re.match(r"POINT\(([0-9.-]+)\s+([0-9.-]+)\)", centroide_ll or "")
    if not match:
        raise ValueError(f"Cannot parse WGS84 point: {centroide_ll}")
    return float(match.group(1)), float(match.group(2))


async def geocode_address(adres: str) -> Optional[dict]:
    """
    Geocode a Dutch address using PDOK Locatieserver.
    Returns dict with x_rd, y_rd, lon, lat, gemeente, gemeente_code, adres_display.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        params = {
            "q": adres,
            "fq": "type:adres",
            "rows": 1,
            "fl": "id,weergavenaam,centroide_ll,centroide_rd,gemeentenaam,gemeentecode",
        }
        response = await client.get(PDOK_FREE_URL, params=params)
        response.raise_for_status()
        data = response.json()

    docs = data.get("response", {}).get("docs", [])
    if not docs:
        return None

    doc = docs[0]
    centroide_rd = doc.get("centroide_rd", "")
    centroide_ll = doc.get("centroide_ll", "")

    try:
        x_rd, y_rd = parse_rd_point(centroide_rd)
    except ValueError:
        return None

    lon, lat = None, None
    try:
        lon, lat = parse_wgs84_point(centroide_ll)
    except ValueError:
        pass

    gemeente_code_raw = doc.get("gemeentecode", "")
    # PDOK returns "0344", DSO expects "gm0344"
    if gemeente_code_raw and not gemeente_code_raw.startswith("gm"):
        gemeente_code = f"gm{gemeente_code_raw}"
    else:
        gemeente_code = gemeente_code_raw

    return {
        "x_rd": x_rd,
        "y_rd": y_rd,
        "lon": lon,
        "lat": lat,
        "gemeente": doc.get("gemeentenaam", ""),
        "gemeente_code": gemeente_code,
        "adres_display": doc.get("weergavenaam", adres),
        "pdok_id": doc.get("id", ""),
    }


async def suggest_address(q: str, rows: int = 8) -> list[dict]:
    """
    Address autocomplete suggestions using PDOK Locatieserver.
    Returns list of {id, weergavenaam} dicts.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        params = {
            "q": q,
            "fq": "type:adres",
            "rows": rows,
            "fl": "id,weergavenaam",
        }
        response = await client.get(PDOK_SUGGEST_URL, params=params)
        response.raise_for_status()
        data = response.json()

    return data.get("response", {}).get("docs", [])
