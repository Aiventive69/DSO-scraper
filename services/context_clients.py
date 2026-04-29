"""
Extra context clients for nationwide datasets beyond plan rules:
- RCE Rijksmonumenten (nearby monument status)
- PDOK BRK Kadastrale Kaart OGC (parcel context)
- PDOK Waterschappen zoneringen OGC (water protection zones)
"""

import asyncio
import logging
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)

RCE_OGC = "https://api.pdok.nl/rce/beschermde-gebieden-cultuurhistorie/ogc/v1"
BRK_OGC = "https://api.pdok.nl/kadaster/brk-kadastrale-kaart/ogc/v1"
WATERSCHAP_OGC = "https://api.pdok.nl/hwh/waterschappen-zoneringen-imwa/ogc/v1"
NATURA2000_OGC = "https://api.pdok.nl/rvo/natura2000/ogc/v1"


def _point_bbox_wgs84(lon: float, lat: float, delta: float = 0.00035) -> str:
    """
    Build a tiny WGS84 bbox around a point.
    delta=0.00035 is roughly 30-40m in NL.
    """
    return f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"


async def fetch_rijksmonumenten_nearby(lon: float, lat: float) -> List[Dict]:
    """
    Fetch nearby Rijksmonumenten via rijksmonumenten.info Solr endpoint.
    Returns a normalized list of monument dicts.
    """
    params = {
        "f": "json",
        "bbox": _point_bbox_wgs84(lon, lat, delta=0.0012),  # ~100m
        "limit": "12",
    }
    url = f"{RCE_OGC}/collections/rce_inspire_points/items"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            docs = resp.json().get("features", [])
    except Exception as e:
        logger.warning(f"RCE monument query failed: {e}")
        return []

    out = []
    for d in docs:
        props = d.get("properties", {})
        citation = props.get("ci_citation", "")
        monument_nr = citation.rstrip("/").split("/")[-1] if citation else props.get("localid", "")
        out.append(
            {
                "monumentnummer": monument_nr,
                "naam": f"Rijksmonument {monument_nr}" if monument_nr else "Rijksmonument",
                "straat": "",
                "woonplaats": "",
                "status": "rijksmonument",
            }
        )
    return out


async def fetch_brk_perceel_context(lon: float, lat: float) -> List[Dict]:
    """
    Fetch nearby BRK parcel features via PDOK OGC API.
    """
    bbox = _point_bbox_wgs84(lon, lat)
    params = {"f": "json", "bbox": bbox, "limit": "8"}
    url = f"{BRK_OGC}/collections/perceel/items"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            feats = resp.json().get("features", [])
    except Exception as e:
        logger.warning(f"BRK perceel query failed: {e}")
        return []

    out = []
    for f in feats:
        p = f.get("properties", {})
        out.append(
            {
                "identificatie": p.get("identificatieLokaalID") or p.get("identificatie") or "",
                "gemeente": p.get("AKRKadastraleGemeenteCodeWaarde")
                or p.get("kadastraleGemeenteWaarde")
                or "",
                "sectie": p.get("sectie")
                or p.get("sectieCode")
                or "",
                "perceelnummer": p.get("perceelnummer")
                or p.get("perceelnummerWaarde")
                or "",
            }
        )
    return out


async def _fetch_waterschap_collection(collection_id: str, lon: float, lat: float) -> List[Dict]:
    bbox = _point_bbox_wgs84(lon, lat, delta=0.0007)  # slightly wider for line/polygon overlaps
    params = {"f": "json", "bbox": bbox, "limit": "10"}
    url = f"{WATERSCHAP_OGC}/collections/{collection_id}/items"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            feats = resp.json().get("features", [])
    except Exception as e:
        logger.warning(f"Waterschap collection '{collection_id}' failed: {e}")
        return []

    out = []
    for f in feats:
        p = f.get("properties", {})
        out.append(
            {
                "type": collection_id,
                "naam": p.get("naam") or p.get("identificatie") or collection_id,
                "beheerder": p.get("beheerder") or p.get("waterbeheerder") or "",
            }
        )
    return out


async def fetch_waterschapszoneringen(lon: float, lat: float) -> List[Dict]:
    """
    Fetch relevant water authority zones around a location.
    """
    collections = [
        "beschermingszone",
        "waterstaatswerkwaterkering",
        "waterstaatswerkwaterbergingsgebied",
        "profielvanvrijeruimte",
    ]
    tasks = [_fetch_waterschap_collection(c, lon, lat) for c in collections]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: List[Dict] = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    return out


async def fetch_natura2000_context(lon: float, lat: float) -> List[Dict]:
    """
    Fetch Natura2000 areas around a location.
    """
    bbox = _point_bbox_wgs84(lon, lat, delta=0.01)  # broader search area (~1km)
    params = {"f": "json", "bbox": bbox, "limit": "8"}
    url = f"{NATURA2000_OGC}/collections/natura2000/items"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            feats = resp.json().get("features", [])
    except Exception as e:
        logger.warning(f"Natura2000 query failed: {e}")
        return []

    out = []
    for f in feats:
        p = f.get("properties", {})
        out.append(
            {
                "naam": p.get("naam") or p.get("name") or p.get("id") or "Natura2000-gebied",
                "code": p.get("code") or p.get("nr") or "",
                "status": p.get("status") or "",
            }
        )
    return out


def format_extra_context_for_llm(
    rijksmonumenten: List[Dict],
    brk_percelen: List[Dict],
    waterschapszones: List[Dict],
    natura2000_gebieden: List[Dict],
) -> str:
    """
    Format extra source context into plain text for LLM consumption.
    """
    lines: List[str] = ["## Aanvullende landelijke bronnen"]

    if rijksmonumenten:
        lines.append("")
        lines.append("### Rijksmonumenten (RCE)")
        for m in rijksmonumenten[:8]:
            naam = m.get("naam", "Onbekend")
            nr = m.get("monumentnummer", "")
            plaats = m.get("woonplaats", "")
            suffix = f" (nr: {nr})" if nr else ""
            lines.append(f"- {naam}{suffix} {plaats}".strip())

    if brk_percelen:
        lines.append("")
        lines.append("### Kadastrale percelen (BRK)")
        for p in brk_percelen[:8]:
            gem = p.get("gemeente", "")
            sec = p.get("sectie", "")
            nr = p.get("perceelnummer", "")
            ident = p.get("identificatie", "")
            lines.append(f"- Gemeente: {gem}, sectie: {sec}, perceel: {nr}, id: {ident}")

    if waterschapszones:
        lines.append("")
        lines.append("### Waterschapszoneringen (IMWA)")
        for z in waterschapszones[:12]:
            ztype = z.get("type", "")
            naam = z.get("naam", "")
            beheerder = z.get("beheerder", "")
            lines.append(f"- {ztype}: {naam} (beheerder: {beheerder})")

    if natura2000_gebieden:
        lines.append("")
        lines.append("### Natura2000 (RVO)")
        for n in natura2000_gebieden[:8]:
            naam = n.get("naam", "Natura2000-gebied")
            code = n.get("code", "")
            status = n.get("status", "")
            lines.append(f"- {naam} (code: {code}, status: {status})")

    if len(lines) == 1:
        return ""
    return "\n".join(lines)
