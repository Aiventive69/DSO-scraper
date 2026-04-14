"""
PDOK Ruimtelijke Plannen WMS client.

Queries the PDOK WMS (GetFeatureInfo) to retrieve bestemmingsplan attributes
for a given RD New coordinate, then fetches the full plan text from
ruimtelijkeplannen.nl.
"""

import re
import logging
import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

PDOK_WMS = "https://service.pdok.nl/kadaster/ruimtelijke-plannen/wms/v1_0"
RP_BASE = "https://ruimtelijkeplannen.nl/documents"

# Layers to query and their human-readable labels
LAYERS = [
    ("plangebied",        "Bestemmingsplan"),
    ("enkelbestemming",   "Bestemming"),
    ("dubbelbestemming",  "Dubbelbestemming"),
    ("bouwvlak",          "Bouwvlak"),
    ("maatvoering",       "Maatvoering"),
    ("functieaanduiding", "Functieaanduiding"),
    ("gebiedsaanduiding", "Gebiedsaanduiding"),
    ("bouwaanduiding",    "Bouwaanduiding"),
]


def _parse_wms_text(text: str) -> Optional[Dict]:
    """Parse plain-text WMS GetFeatureInfo response into a dict."""
    if not text or "no results" in text.lower() or "Search returned no results" in text:
        return None
    props: Dict = {}
    for line in text.splitlines():
        m = re.match(r"\s+(\w[\w\s\-]+?)\s+=\s+'?([^']*)'?\s*$", line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            if val:
                props[key] = val
    return props if props else None


async def _fetch_layer(client: httpx.AsyncClient, layer: str, bbox: str) -> Tuple[str, Optional[Dict]]:
    """Fetch a single WMS layer (used for parallel execution)."""
    try:
        r = await client.get(PDOK_WMS, params={
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetFeatureInfo",
            "LAYERS": layer,
            "QUERY_LAYERS": layer,
            "FORMAT": "image/png",
            "INFO_FORMAT": "text/plain",
            "WIDTH": "256",
            "HEIGHT": "256",
            "CRS": "EPSG:28992",
            "BBOX": bbox,
            "I": "128",
            "J": "128",
        })
        r.raise_for_status()
        props = _parse_wms_text(r.text)
        if props:
            logger.info(f"WMS {layer}: naam={props.get('naam', '')}")
        return layer, props
    except Exception as e:
        logger.warning(f"WMS layer {layer} error: {e}")
        return layer, None


async def get_bestemmingsplan_features(x_rd: float, y_rd: float) -> Dict[str, Optional[Dict]]:
    """
    Query all relevant WMS layers for the given RD coordinate (in parallel).
    Returns a dict keyed by layer name containing parsed feature properties.
    """
    import asyncio
    delta = 500  # 500m bounding box
    bbox = f"{x_rd - delta},{y_rd - delta},{x_rd + delta},{y_rd + delta}"

    async with httpx.AsyncClient(timeout=20) as client:
        tasks = [_fetch_layer(client, layer, bbox) for layer, _ in LAYERS]
        pairs = await asyncio.gather(*tasks)

    return dict(pairs)


async def fetch_full_plan_text(url: str) -> str:
    """
    Fetch the complete plan rules document from ruimtelijkeplannen.nl.
    Supports both HTML and PDF documents. Returns full plain text.
    """
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
    except Exception as e:
        logger.warning(f"Plan text fetch error ({url}): {e}")
        return ""

    content_type = r.headers.get("content-type", "")

    # PDF document
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return _extract_pdf_text(r.content)

    # HTML document
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf_text(content: bytes) -> str:
    """Extract plain text from a PDF file (bytes)."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
        text = "\n\n".join(parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        logger.info(f"PDF gelezen: {len(reader.pages)} pagina's, {len(text)} tekens")
        return text.strip()
    except Exception as e:
        logger.warning(f"PDF extractie mislukt: {e}")
        return ""


async def fetch_plan_text(url: str, anchor: Optional[str] = None) -> str:
    """
    Fetch HTML plan text from ruimtelijkeplannen.nl and return the
    relevant section as clean plain text.
    If anchor is provided, only the section starting at that anchor is returned.
    """
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
    except Exception as e:
        logger.warning(f"Plan text fetch error ({url}): {e}")
        return ""

    soup = BeautifulSoup(r.text, "html.parser")

    if anchor:
        start = soup.find(id=anchor) or soup.find("a", {"name": anchor})
        if start:
            parts = []
            el = start
            count = 0
            while el and count < 80:
                txt = el.get_text(separator=" ", strip=True)
                if txt:
                    parts.append(txt)
                el = el.find_next_sibling()
                count += 1
                if el and el.name in ("h1", "h2") and count > 3:
                    break
            return " ".join(parts)

    return soup.get_text(separator="\n", strip=True)


def _extract_anchor_from_url(url: str) -> Tuple[str, Optional[str]]:
    """Split a URL into base URL and anchor fragment."""
    if "#" in url:
        base, anchor = url.split("#", 1)
        return base, anchor
    return url, None


def _extract_maatvoering(props: Dict) -> str:
    """Parse maatvoering field into a human-readable string."""
    raw = props.get("maatvoering", "")
    if not raw:
        return ""
    # Format: '"key"="value"' possibly repeated
    items = re.findall(r'"([^"]+)"\s*=\s*"([^"]+)"', raw)
    if items:
        return "; ".join(f"{k}: {v}" for k, v in items)
    return raw


def _pick_regels_url(links: List[str]) -> Optional[str]:
    """
    Select the best 'rules' document URL from a list of plan document links.
    Priority order:
      1. r_*.html  — IMRO2012 planregels (HTML, best)
      2. v_*.pdf   — IMRO2008 voorschriften (PDF)
      3. r_*.pdf   — planregels als PDF
    """
    r_html = next((l for l in links if re.search(r'/r_[^/]+\.html$', l)), None)
    if r_html:
        return r_html
    v_pdf = next((l for l in links if re.search(r'/v_[^/]+\.pdf$', l)), None)
    if v_pdf:
        return v_pdf
    r_pdf = next((l for l in links if re.search(r'/r_[^/]+\.pdf$', l)), None)
    return r_pdf


async def get_bestemmingsplan_data(x_rd: float, y_rd: float) -> Dict:
    """
    Full pipeline: WMS feature info + plan text fetching (all in parallel).
    Returns a structured dict with all relevant plan information.
    """
    import asyncio

    features = await get_bestemmingsplan_features(x_rd, y_rd)

    result: Dict = {
        "plan_naam": None,
        "plan_id": None,
        "plan_status": None,
        "plan_datum": None,
        "plan_links": [],
        "bestemming": None,
        "bestemming_url": None,
        "volledige_regels_tekst": "",
        "dubbelbestemmingen": [],
        "gebiedsaanduidingen": [],
        "functieaanduidingen": [],
        "bouwaanduidingen": [],
        "bouwvlak": False,
        "maatvoering": [],
        "raw_features": features,
    }

    # --- Plangebied ---
    pg = features.get("plangebied")
    if pg:
        result["plan_naam"] = pg.get("naam")
        result["plan_id"] = pg.get("identificatie")
        result["plan_status"] = pg.get("planstatus")
        result["plan_datum"] = pg.get("datum")
        links_raw = pg.get("verwijzingnaartekst", "")
        result["plan_links"] = [ln.strip() for ln in links_raw.split(",") if ln.strip()]

    # --- Bouwvlak ---
    result["bouwvlak"] = features.get("bouwvlak") is not None

    # --- Maatvoering ---
    mv = features.get("maatvoering")
    if mv:
        mv_str = _extract_maatvoering(mv)
        if mv_str:
            result["maatvoering"].append(mv_str)

    # --- Functieaanduiding ---
    fa = features.get("functieaanduiding")
    if fa and fa.get("naam"):
        result["functieaanduidingen"].append(fa["naam"])

    # --- Bouwaanduiding ---
    ba = features.get("bouwaanduiding")
    if ba and ba.get("naam"):
        result["bouwaanduidingen"].append(ba["naam"])

    # --- Enkelbestemming / Dubbelbestemming / Gebiedsaanduiding namen ---
    eb = features.get("enkelbestemming")
    if eb:
        result["bestemming"] = eb.get("naam")
        url_raw = eb.get("verwijzingnaartekst", "")
        if url_raw:
            result["bestemming_url"] = url_raw.strip().split("#")[0]

    db = features.get("dubbelbestemming")
    if db and db.get("naam"):
        result["dubbelbestemmingen"].append(db["naam"])

    ga = features.get("gebiedsaanduiding")
    if ga and ga.get("naam"):
        result["gebiedsaanduidingen"].append(ga["naam"])

    # --- Haal het VOLLEDIGE regelsdocument op ---
    # Prioriteit: r_ (HTML regels, IMRO2012) → v_ (PDF voorschriften, IMRO2008) → t_ (toelichting)
    regels_url = _pick_regels_url(result["plan_links"])

    if regels_url:
        logger.info(f"Ophalen regelsdocument: {regels_url}")
        result["volledige_regels_tekst"] = await fetch_full_plan_text(regels_url)
        logger.info(f"Regelsdocument: {len(result['volledige_regels_tekst'])} tekens")
    elif result.get("bestemming_url"):
        logger.info(f"Geen regelsdocument gevonden, fallback naar bestemming URL")
        result["volledige_regels_tekst"] = await fetch_full_plan_text(
            result["bestemming_url"].split("#")[0]
        )

    return result


def format_bestemmingsplan_for_llm(data: Dict) -> str:
    """Format the retrieved bestemmingsplan data as plain text for the LLM."""
    lines: List[str] = []

    # Metadata header
    if data.get("plan_naam"):
        lines.append(f"## Bestemmingsplan: {data['plan_naam']}")
        if data.get("plan_id"):
            lines.append(f"Identificatie: {data['plan_id']}")
        if data.get("plan_datum"):
            lines.append(f"Datum: {data['plan_datum']}")
        if data.get("plan_status"):
            lines.append(f"Status: {data['plan_status']}")
        lines.append("")

    # Perceel-specifieke kenmerken
    if data.get("bestemming"):
        lines.append(f"Bestemming op dit perceel: {data['bestemming']}")
    if data.get("bouwvlak"):
        lines.append("Bouwvlak aanwezig: ja")
    if data.get("maatvoering"):
        lines.append(f"Maatvoering: {'; '.join(data['maatvoering'])}")
    if data.get("dubbelbestemmingen"):
        lines.append(f"Dubbelbestemming(en): {', '.join(data['dubbelbestemmingen'])}")
    if data.get("gebiedsaanduidingen"):
        lines.append(f"Gebiedsaanduiding(en): {', '.join(data['gebiedsaanduidingen'])}")
    if data.get("functieaanduidingen"):
        lines.append(f"Functieaanduiding(en): {', '.join(data['functieaanduidingen'])}")
    if data.get("bouwaanduidingen"):
        lines.append(f"Bouwaanduiding(en): {', '.join(data['bouwaanduidingen'])}")
    lines.append("")

    # Volledige regelstekst
    if data.get("volledige_regels_tekst"):
        lines.append("## Volledige planregels")
        lines.append(data["volledige_regels_tekst"])

    return "\n".join(lines)
