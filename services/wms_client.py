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


def _parse_wms_text_multi(text: str) -> List[Dict]:
    """
    Parse a WMS GetFeatureInfo plain-text response that may contain
    multiple features (when FEATURE_COUNT > 1).
    Returns a list of property dicts, one per feature.
    """
    if not text or "no results" in text.lower() or "Search returned no results" in text:
        return []
    features = []
    current: Dict = {}
    for line in text.splitlines():
        # New feature block
        if re.match(r"\s*Feature \d+:", line):
            if current:
                features.append(current)
            current = {}
            continue
        m = re.match(r"\s+(\w[\w\s\-]+?)\s+=\s+'?([^']*)'?\s*$", line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            if val:
                current[key] = val
    if current:
        features.append(current)
    return features


async def _fetch_layer(client: httpx.AsyncClient, layer: str, bbox: str, feature_count: int = 1) -> Tuple[str, Optional[Dict]]:
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
            "FEATURE_COUNT": str(feature_count),
        })
        r.raise_for_status()
        props = _parse_wms_text(r.text)
        if props:
            logger.info(f"WMS {layer}: naam={props.get('naam', '')}")
        return layer, props
    except Exception as e:
        logger.warning(f"WMS layer {layer} error: {e}")
        return layer, None


async def _fetch_layer_multi(client: httpx.AsyncClient, layer: str, bbox: str, feature_count: int = 10) -> Tuple[str, List[Dict]]:
    """Fetch a WMS layer and return ALL matching features (for maatvoering etc.)."""
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
            "FEATURE_COUNT": str(feature_count),
        })
        r.raise_for_status()
        feats = _parse_wms_text_multi(r.text)
        logger.info(f"WMS {layer} (multi): {len(feats)} features")
        return layer, feats
    except Exception as e:
        logger.warning(f"WMS layer {layer} (multi) error: {e}")
        return layer, []


async def _fetch_enkelbestemming_at_point(x_rd: float, y_rd: float, delta: float = 120.0) -> Optional[Dict]:
    """
    Fetch enkelbestemming at a specific point.
    Used for robust bestemming selection via multi-point sampling.
    """
    bbox = f"{x_rd - delta},{y_rd - delta},{x_rd + delta},{y_rd + delta}"
    async with httpx.AsyncClient(timeout=20) as client:
        _, props = await _fetch_layer(client, "enkelbestemming", bbox, feature_count=1)
        return props


async def _pick_bestemming_by_sampling(
    x_rd: float,
    y_rd: float,
    preferred_points: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    """
    Query enkelbestemming on center + nearby offsets and choose the most frequent result.
    This reduces single-pixel misclassification on parcel boundaries/road edges.
    """
    import asyncio
    points: List[tuple] = []
    if preferred_points:
        for p in preferred_points[:16]:
            try:
                points.append((float(p["x"]), float(p["y"])))
            except Exception:
                continue
    # add local fallback points around center
    offsets = [(0.0, 0.0), (6.0, 0.0), (-6.0, 0.0), (0.0, 6.0), (0.0, -6.0)]
    points.extend([(x_rd + dx, y_rd + dy) for dx, dy in offsets])
    tasks = [_fetch_enkelbestemming_at_point(px, py) for px, py in points]
    samples = await asyncio.gather(*tasks)
    valid = [s for s in samples if s and s.get("naam")]
    if not valid:
        return None

    # Majority vote on 'naam'
    counts: Dict[str, int] = {}
    for v in valid:
        nm = v.get("naam", "")
        counts[nm] = counts.get(nm, 0) + 1
    best_name = max(counts.items(), key=lambda kv: kv[1])[0]
    # Return first matching sample as representative
    for v in valid:
        if v.get("naam") == best_name:
            return v
    return valid[0]


async def _find_nearest_enkelbestemming(x_rd: float, y_rd: float) -> Optional[Dict]:
    """
    Fallback search for enkelbestemming around the address point.
    Useful when the exact geocode point falls on road/overlay layers.
    """
    import asyncio
    radii = [0.0, 8.0, 15.0, 25.0, 40.0, 60.0, 90.0, 130.0, 180.0]
    for r in radii:
        if r == 0:
            points = [(x_rd, y_rd)]
        else:
            points = [
                (x_rd + r, y_rd), (x_rd - r, y_rd), (x_rd, y_rd + r), (x_rd, y_rd - r),
                (x_rd + r, y_rd + r), (x_rd - r, y_rd - r), (x_rd + r, y_rd - r), (x_rd - r, y_rd + r),
            ]
        tasks = [_fetch_enkelbestemming_at_point(px, py) for px, py in points]
        samples = await asyncio.gather(*tasks)
        valid = [s for s in samples if s and s.get("naam")]
        if valid:
            # Use majority vote within this radius ring
            counts: Dict[str, int] = {}
            for v in valid:
                nm = v.get("naam", "")
                counts[nm] = counts.get(nm, 0) + 1
            best_name = max(counts.items(), key=lambda kv: kv[1])[0]
            for v in valid:
                if v.get("naam") == best_name:
                    logger.info(f"Bestemming gevonden op radius {r}m: {best_name}")
                    return v
            return valid[0]
    return None


async def get_bestemmingsplan_features(x_rd: float, y_rd: float) -> Dict:
    """
    Query all relevant WMS layers for the given RD coordinate (in parallel).
    Returns a dict keyed by layer name.
    Layers that can have multiple overlapping features (maatvoering, bouwaanduiding,
    gebiedsaanduiding) return a list; others return a single dict or None.
    """
    import asyncio
    delta = 500  # 500m bounding box
    bbox = f"{x_rd - delta},{y_rd - delta},{x_rd + delta},{y_rd + delta}"

    # Layers that return a single feature
    single_layers = ["plangebied", "enkelbestemming", "dubbelbestemming", "bouwvlak", "functieaanduiding"]
    # Layers where multiple overlapping features are common
    multi_layers = ["maatvoering", "bouwaanduiding", "gebiedsaanduiding"]

    async with httpx.AsyncClient(timeout=20) as client:
        single_tasks = [_fetch_layer(client, layer, bbox) for layer in single_layers]
        multi_tasks = [_fetch_layer_multi(client, layer, bbox) for layer in multi_layers]
        single_results = await asyncio.gather(*single_tasks)
        multi_results = await asyncio.gather(*multi_tasks)

    result: Dict = {}
    for layer_name, props in single_results:
        result[layer_name] = props
    for layer_name, props_list in multi_results:
        result[layer_name] = props_list
    return result


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
    If anchor is provided, extract the full article section for that anchor.
    """
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
    except Exception as e:
        logger.warning(f"Plan text fetch error ({url}): {e}")
        return ""

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    if anchor:
        # Find the anchor element (id or <a name="...">)
        start_el = soup.find(id=anchor) or soup.find("a", {"name": anchor})
        if start_el:
            # Walk upward to find the container (div/section/article) for this anchor
            container = start_el
            for _ in range(5):
                parent = container.parent
                if parent and parent.name in ("div", "section", "article", "li"):
                    container = parent
                else:
                    break

            parts = [container.get_text(separator="\n", strip=True)]

            # Also collect following siblings until the next top-level article heading
            el = container.find_next_sibling()
            count = 0
            while el and count < 150:
                tag_name = el.name or ""
                text = el.get_text(separator="\n", strip=True)
                # Stop at a same-level or higher article heading (h2/h3 sibling that isn't an anchor)
                if tag_name in ("h1", "h2") and count > 0:
                    break
                if tag_name == "h3" and count > 5 and text and re.match(r"^(Artikel|Hoofdstuk|Afdeling)\s+\d", text):
                    break
                if text:
                    parts.append(text)
                el = el.find_next_sibling()
                count += 1

            text = "\n\n".join(parts)
            text = re.sub(r"\n{3,}", "\n\n", text)
            logger.info(f"Anchor-tekst '{anchor}': {len(text)} tekens")
            return text.strip()

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


def _pick_bijlagen_urls(links: List[str]) -> List[str]:
    """
    Pick likely annex/list documents (bijlagen), e.g. b_*.html/pdf.
    These often contain 'Lijst van bedrijfsactiviteiten'.
    """
    candidates = [l for l in links if re.search(r'/b_[^/]+\.(html|pdf)$', l, re.IGNORECASE)]
    # Keep deterministic order and avoid huge fan-out
    return candidates[:2]


async def get_bestemmingsplan_data(x_rd: float, y_rd: float, bag_sampling_points: Optional[List[Dict]] = None) -> Dict:
    """
    Full pipeline: WMS feature info + plan text fetching (all in parallel).

    Fetching strategy:
    - Specifieke bestemmingsartikel via anchor URL → bevat kap/bouwregels voor DEZE bestemming
    - Volledig regelsdocument (eerste deel) → bevat algemene regels, parkeren, gebruik etc.
    Both are stored separately so the LLM always gets the focused article text.
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
        "bestemming_artikel_tekst": "",   # Specific bestemming article (via anchor)
        "algemene_regels_tekst": "",       # Full document text for general provisions
        "bijlagen_tekst": "",              # Annex text (e.g. lijst bedrijfsactiviteiten)
        "bestemming_bron": "",             # direct | sampled | nearby | onbekend
        "heeft_perceel_specifieke_regels": False,
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

    # --- Maatvoering (multi-feature list) ---
    mv_list = features.get("maatvoering") or []
    if isinstance(mv_list, dict):
        mv_list = [mv_list]
    for mv in mv_list:
        mv_str = _extract_maatvoering(mv)
        if mv_str:
            result["maatvoering"].append(mv_str)

    # --- Functieaanduiding ---
    fa = features.get("functieaanduiding")
    if isinstance(fa, list):
        fa = fa[0] if fa else None
    if fa and fa.get("naam"):
        result["functieaanduidingen"].append(fa["naam"])

    # --- Bouwaanduiding (multi-feature list) ---
    ba_list = features.get("bouwaanduiding") or []
    if isinstance(ba_list, dict):
        ba_list = [ba_list]
    for ba in ba_list:
        if ba.get("naam"):
            result["bouwaanduidingen"].append(ba["naam"])

    # --- Enkelbestemming / Dubbelbestemming / Gebiedsaanduiding ---
    eb = features.get("enkelbestemming")
    bestemming_anchor_url = None
    if eb:
        result["bestemming"] = eb.get("naam")
        result["bestemming_bron"] = "direct"
        result["heeft_perceel_specifieke_regels"] = True
        url_raw = eb.get("verwijzingnaartekst", "")
        if url_raw:
            full_url = url_raw.strip()
            base_url = full_url.split("#")[0]
            result["bestemming_url"] = base_url
            # Keep the full anchor URL for precise article fetching
            bestemming_anchor_url = full_url if "#" in full_url else None

    # Robust bestemming correction via multi-point sampling
    sampled_eb = await _pick_bestemming_by_sampling(
        x_rd=x_rd,
        y_rd=y_rd,
        preferred_points=bag_sampling_points,
    )
    if sampled_eb and sampled_eb.get("naam"):
        sampled_name = sampled_eb.get("naam")
        if sampled_name != result.get("bestemming"):
            logger.info(
                f"Bestemming gecorrigeerd via sampling: "
                f"'{result.get('bestemming')}' -> '{sampled_name}'"
            )
        result["bestemming"] = sampled_name
        result["bestemming_bron"] = "sampled_bag" if bag_sampling_points else "sampled"
        result["heeft_perceel_specifieke_regels"] = True
        sampled_url_raw = sampled_eb.get("verwijzingnaartekst", "")
        if sampled_url_raw:
            sampled_full = sampled_url_raw.strip()
            result["bestemming_url"] = sampled_full.split("#")[0]
            bestemming_anchor_url = sampled_full if "#" in sampled_full else bestemming_anchor_url

    # Final fallback: search nearby rings if still no bestemming
    if not result.get("bestemming"):
        nearby_eb = await _find_nearest_enkelbestemming(x_rd, y_rd)
        if nearby_eb and nearby_eb.get("naam"):
            result["bestemming"] = nearby_eb.get("naam")
            result["bestemming_bron"] = "nearby"
            result["heeft_perceel_specifieke_regels"] = False
            nearby_url_raw = nearby_eb.get("verwijzingnaartekst", "")
            if nearby_url_raw:
                nearby_full = nearby_url_raw.strip()
                result["bestemming_url"] = nearby_full.split("#")[0]
                bestemming_anchor_url = nearby_full if "#" in nearby_full else bestemming_anchor_url

    db = features.get("dubbelbestemming")
    if isinstance(db, list):
        db = db[0] if db else None
    if db and db.get("naam"):
        result["dubbelbestemmingen"].append(db["naam"])

    # --- Gebiedsaanduiding (multi-feature list) ---
    ga_list = features.get("gebiedsaanduiding") or []
    if isinstance(ga_list, dict):
        ga_list = [ga_list]
    for ga in ga_list:
        if ga.get("naam"):
            result["gebiedsaanduidingen"].append(ga["naam"])

    # --- Fetch plan texts in parallel ---
    # 1. Article-specific text (via anchor) → always contains kap/bouw rules for THIS bestemming
    # 2. Full document (for general rules like parking, permitted use, etc.)
    regels_url = _pick_regels_url(result["plan_links"])

    fetch_tasks = []
    task_keys: List[str] = []

    if bestemming_anchor_url:
        base, anchor = _extract_anchor_from_url(bestemming_anchor_url)
        fetch_tasks.append(fetch_plan_text(base, anchor))
        task_keys.append("artikel")

    if regels_url:
        fetch_tasks.append(fetch_full_plan_text(regels_url))
        task_keys.append("volledig")
    elif result.get("bestemming_url"):
        fetch_tasks.append(fetch_full_plan_text(result["bestemming_url"]))
        task_keys.append("volledig")

    # Fetch annex documents (bijlagen), often containing business category lists
    for b_url in _pick_bijlagen_urls(result.get("plan_links", [])):
        fetch_tasks.append(fetch_full_plan_text(b_url))
        task_keys.append("bijlage")

    if fetch_tasks:
        texts = await asyncio.gather(*fetch_tasks)
        for key, tekst in zip(task_keys, texts):
            if key == "artikel":
                result["bestemming_artikel_tekst"] = tekst
                logger.info(f"Bestemmingsartikel: {len(tekst)} tekens")
            elif key == "volledig":
                result["algemene_regels_tekst"] = tekst
                logger.info(f"Volledig document: {len(tekst)} tekens")
            elif key == "bijlage" and tekst:
                if result["bijlagen_tekst"]:
                    result["bijlagen_tekst"] += "\n\n--- BIJLAGE ---\n\n" + tekst
                else:
                    result["bijlagen_tekst"] = tekst
                logger.info(f"Bijlage toegevoegd: {len(tekst)} tekens")

    return result


def format_bestemmingsplan_for_llm(data: Dict) -> str:
    """
    Format bestemmingsplan data for the LLM.

    Layout (priority order):
    1. Metadata + perceel-specifieke kenmerken (WMS attributes)
    2. VOLLEDIG bestemmingsartikel voor deze bestemming (via anchor) — altijd volledig
    3. Overige planregels / algemene bepalingen (eerste N tekens van volledig document)

    This ensures kap/bouwregels from the specific article are ALWAYS present,
    even if the article appears late in a large document.
    """
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

    # Perceel-specifieke kenmerken (from WMS map attributes)
    if data.get("bestemming"):
        bron = data.get("bestemming_bron", "onbekend")
        if bron == "nearby":
            lines.append(f"**Waarschijnlijke bestemming nabij dit perceel: {data['bestemming']}**")
            lines.append("_(Afgeleid uit nabijgelegen kaartobjecten; verifieer in 'Regels op de kaart')_")
        elif bron == "sampled_bag":
            lines.append(f"**Bestemming op dit perceel: {data['bestemming']}**")
            lines.append("_(Afgeleid uit sampling op BAG pandgeometrie)_")
        else:
            lines.append(f"**Bestemming op dit perceel: {data['bestemming']}**")
    else:
        lines.append("**Bestemming op dit perceel: ONBEKEND (kaartlaag gaf geen direct resultaat)**")
    if data.get("bouwvlak"):
        lines.append("Bouwvlak aanwezig: ja")
    if data.get("maatvoering"):
        lines.append(f"Maatvoering (kaart): {'; '.join(data['maatvoering'])}")
    if data.get("dubbelbestemmingen"):
        lines.append(f"Dubbelbestemming(en): {', '.join(data['dubbelbestemmingen'])}")
    if data.get("gebiedsaanduidingen"):
        lines.append(f"Gebiedsaanduiding(en): {', '.join(data['gebiedsaanduidingen'])}")
    if data.get("functieaanduidingen"):
        lines.append(f"Functieaanduiding(en): {', '.join(data['functieaanduidingen'])}")
    if data.get("bouwaanduidingen"):
        lines.append(f"Bouwaanduiding(en): {', '.join(data['bouwaanduidingen'])}")
    lines.append("")

    # Specific bestemming article (via anchor URL — contains all rules for this bestemming)
    artikel_tekst = data.get("bestemming_artikel_tekst", "").strip()
    if data.get("heeft_perceel_specifieke_regels"):
        lines.append("**Status perceelregels: PERCEEL-SPECIFIEK GEVONDEN**")
    else:
        lines.append("**Status perceelregels: GEEN ZEKERE PERCEEL-SPECIFIEKE ENKELBESTEMMING GEVONDEN**")
        lines.append(
            "Gebruik onderstaande planregels als algemene context voor de locatie. "
            "Bevestig perceelgerichte uitkomsten in 'Regels op de kaart'."
        )
    lines.append("")

    if artikel_tekst:
        lines.append(f"## Bestemmingsartikel: {data.get('bestemming', 'bestemming')}")
        lines.append(artikel_tekst)
        lines.append("")

    # Full document text — no truncation needed; the summarizer handles chunking
    algemeen_tekst = data.get("algemene_regels_tekst", "").strip()
    if algemeen_tekst:
        lines.append("## Volledige planregels")
        lines.append(algemeen_tekst)
    elif not artikel_tekst:
        legacy = data.get("volledige_regels_tekst", "").strip()
        if legacy:
            lines.append("## Planregels")
            lines.append(legacy)

    # Annexes often include concrete lists like 'Lijst van bedrijfsactiviteiten'
    if data.get("bijlagen_tekst"):
        lines.append("")
        lines.append("## Bijlagen / Lijsten")
        lines.append(data["bijlagen_tekst"])

    return "\n".join(lines)
