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


def _looks_like_address_with_number(adres: str) -> bool:
    """Check whether the query contains a house number."""
    return bool(re.search(r"\b\d+\b", adres))


async def _geocode_query(client: httpx.AsyncClient, q: str, type_filter: Optional[str] = None) -> list:
    """Internal helper: geocode with optional PDOK type filter. Returns list of docs."""
    params: dict = {
        "q": q,
        "rows": 3,
        "fl": "id,weergavenaam,type,centroide_ll,centroide_rd,gemeentenaam,gemeentecode",
    }
    if type_filter:
        params["fq"] = f"type:{type_filter}"
    r = await client.get(PDOK_FREE_URL, params=params)
    r.raise_for_status()
    return r.json().get("response", {}).get("docs", [])


def _result_is_relevant(query: str, weergavenaam: str) -> bool:
    """
    Check if the geocoded result is plausibly related to the query.
    Requires that a STREET NAME word from the query matches the result —
    city/municipality names alone are not sufficient.
    """
    # Words that don't indicate a street name match
    generic_stopwords = {
        "de", "den", "het", "van", "aan", "te", "in", "op", "bij",
        "straat", "weg", "laan", "plein", "dijk", "kade", "gracht",
        "singel", "baan", "pad", "steeg", "dwars", "straat",
        # Common Dutch city names that should NOT count as a street match
        "amsterdam", "rotterdam", "utrecht", "eindhoven", "groningen",
        "breda", "tilburg", "nijmegen", "enschede", "haarlem",
        "arnhem", "zaandam", "leiden", "dordrecht", "zoetermeer",
        "zwolle", "maastricht", "delft", "alkmaar", "apeldoorn",
        "venlo", "deventer", "amersfoort", "leeuwarden", "hilversum",
    }
    query_words = set(re.sub(r"[^a-zA-Z ]", " ", query.lower()).split()) - generic_stopwords
    result_words = set(re.sub(r"[^a-zA-Z ]", " ", weergavenaam.lower()).split()) - generic_stopwords

    if not query_words:
        return True

    for qw in query_words:
        if len(qw) < 4:
            continue  # Skip very short words
        for rw in result_words:
            if len(rw) < 4:
                continue
            if qw in rw or rw in qw:
                return True
    return False


async def geocode_address(adres: str) -> Optional[dict]:
    """
    Geocode a Dutch address using PDOK Locatieserver.
    Returns dict with x_rd, y_rd, lon, lat, gemeente, gemeente_code, adres_display.

    Strategy:
    1. Try type:adres — best for "Straatnaam 10, Stad"
    2. Validate relevance; if unrelated, discard
    3. Fallback: no type filter (catches type:weg for street-only queries like "Juffersstraat Utrecht")
    4. Returns None only if nothing useful found.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        doc = None

        # Step 1: exact address lookup
        adres_docs = await _geocode_query(client, adres, "adres")
        for candidate in adres_docs:
            if _result_is_relevant(adres, candidate.get("weergavenaam", "")):
                doc = candidate
                break

        # Step 2: no type filter — finds streets (type:weg), postcodes, etc.
        if not doc:
            all_docs = await _geocode_query(client, adres, None)
            for candidate in all_docs:
                if _result_is_relevant(adres, candidate.get("weergavenaam", "")):
                    doc = candidate
                    break

    if not doc:
        return None

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
    if gemeente_code_raw and not gemeente_code_raw.startswith("gm"):
        gemeente_code = f"gm{gemeente_code_raw}"
    else:
        gemeente_code = gemeente_code_raw

    geocode_type = doc.get("type", "adres")
    is_exact = geocode_type == "adres"

    return {
        "x_rd": x_rd,
        "y_rd": y_rd,
        "lon": lon,
        "lat": lat,
        "gemeente": doc.get("gemeentenaam", ""),
        "gemeente_code": gemeente_code,
        "adres_display": doc.get("weergavenaam", adres),
        "pdok_id": doc.get("id", ""),
        "heeft_huisnummer": _looks_like_address_with_number(adres),
        "is_exact_adres": is_exact,
        "geocode_type": geocode_type,
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
