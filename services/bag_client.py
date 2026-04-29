"""
Kadaster BAG API client (Individuele Bevragingen v2).
Provides robust object-identification context for an address.
"""

import logging
import math
import re
from typing import Dict, Optional, Tuple, List, Any

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
    bag_data = {
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
    bag_data["rd_points"] = await _fetch_pand_sampling_points(
        pand_ids=bag_data["pand_ids"],
        bag_api_key=bag_api_key,
    )
    return bag_data


def _find_coordinates_recursive(obj: Any) -> Optional[List]:
    """Find first GeoJSON-like coordinates list in nested object."""
    if isinstance(obj, dict):
        if "coordinates" in obj and isinstance(obj["coordinates"], list):
            return obj["coordinates"]
        for v in obj.values():
            found = _find_coordinates_recursive(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_coordinates_recursive(v)
            if found is not None:
                return found
    return None


def _flatten_ring_points(coords: List) -> List[Tuple[float, float]]:
    """
    Convert likely polygon/multipolygon coordinates to a flat list of (x,y) points.
    Supports:
    - Polygon: [[[x,y], ...]]
    - MultiPolygon: [[[[x,y], ...]], ...]
    """
    points: List[Tuple[float, float]] = []
    if not coords:
        return points
    # Polygon outer ring
    if isinstance(coords[0], list) and coords and coords[0] and isinstance(coords[0][0], (int, float)):
        # LineString style
        for p in coords:
            if isinstance(p, list) and len(p) >= 2:
                points.append((float(p[0]), float(p[1])))
        return points
    if isinstance(coords[0], list) and coords[0] and isinstance(coords[0][0], list):
        # Could be polygon or multipolygon
        # Polygon: coords[0] is ring of points
        if coords[0] and coords[0][0] and isinstance(coords[0][0][0], (int, float)):
            for p in coords[0]:
                if isinstance(p, list) and len(p) >= 2:
                    points.append((float(p[0]), float(p[1])))
            return points
        # MultiPolygon: coords[0][0] is ring
        if coords[0] and coords[0][0] and coords[0][0][0] and isinstance(coords[0][0][0][0], (int, float)):
            for poly in coords:
                if not poly or not poly[0]:
                    continue
                for p in poly[0]:
                    if isinstance(p, list) and len(p) >= 2:
                        points.append((float(p[0]), float(p[1])))
            return points
    return points


def _build_sampling_points(points: List[Tuple[float, float]], max_points: int = 12) -> List[Dict]:
    """Build representative sampling points from polygon vertices + centroid."""
    if not points:
        return []
    # centroid (simple average; good enough as sampling point)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    result = [{"x": cx, "y": cy}]

    # add a spread of vertices around ring
    step = max(1, math.ceil(len(points) / (max_points - 1)))
    for i in range(0, len(points), step):
        x, y = points[i]
        result.append({"x": x, "y": y})
        if len(result) >= max_points:
            break
    return result


async def _fetch_pand_sampling_points(pand_ids: List[str], bag_api_key: str) -> List[Dict]:
    """
    Fetch first pand geometry and create representative RD points for bestemming sampling.
    """
    if not pand_ids:
        return []

    headers = {
        "X-Api-Key": bag_api_key,
        "Accept": "application/json",
        "Accept-Crs": "epsg:28992",
    }
    pand_id = pand_ids[0]
    url = f"{BAG_BASE}/panden/{pand_id}"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"BAG pand geometry query failed: {e}")
        return []

    coords = _find_coordinates_recursive(data)
    if not coords:
        return []
    ring_points = _flatten_ring_points(coords)
    return _build_sampling_points(ring_points)


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
    rd_points = bag_data.get("rd_points", [])
    if rd_points:
        lines.append(f"Pand geometrie samplingpunten: {len(rd_points)}")
    return "\n".join(lines)
