"""
DSO Omgevingsdocumenten Presenteren API v8 client.
Queries omgevingsplannen and their rules for a given location.
"""

import re
import httpx
from bs4 import BeautifulSoup
from typing import Optional

CRS_RD = "http://www.opengis.net/def/crs/EPSG/0/28992"


def to_uri_identificatie(identificatie: str) -> str:
    """
    Convert DSO AKN identificatie to URL-safe uriIdentificatie.
    e.g. /akn/nl/act/gm0363/2020/omgevingsplan -> _akn_nl_act_gm0363_2020_omgevingsplan
    """
    return identificatie.replace("/", "_").replace("-", "_")


def _dso_headers(api_key: str) -> dict:
    return {
        "X-Api-Key": api_key,
        "Accept": "application/hal+json",
        "Content-Type": "application/json",
        "Content-Crs": CRS_RD,
    }


def _point_geometry(x_rd: float, y_rd: float) -> dict:
    return {"type": "Point", "coordinates": [x_rd, y_rd]}


def clean_xhtml(xhtml: str) -> str:
    """Strip XHTML tags from DSO text content, returning clean plain text."""
    if not xhtml:
        return ""
    soup = BeautifulSoup(xhtml, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


class DSOClient:
    def __init__(self, api_key: str, use_production: bool = False):
        self.api_key = api_key
        if use_production:
            self.base_url = "https://service.omgevingswet.overheid.nl/publiek/omgevingsdocumenten/api/presenteren/v8"
        else:
            self.base_url = "https://service.pre.omgevingswet.overheid.nl/publiek/omgevingsdocumenten/api/presenteren/v8"

    async def zoek_regelingen(
        self,
        x_rd: float,
        y_rd: float,
        gemeente_code: Optional[str] = None,
        type_bevoegd_gezag: Optional[list] = None,
    ) -> list[dict]:
        """
        Search for regelingen (incl. omgevingsplannen) at a given RD New location.
        Returns list of regelingen sorted by relevance.
        """
        body: dict = {
            "geometrie": _point_geometry(x_rd, y_rd),
        }
        if type_bevoegd_gezag:
            body["typeBevoegdGezag"] = type_bevoegd_gezag
        if gemeente_code:
            body["bevoegdGezag"] = [gemeente_code]

        headers = _dso_headers(self.api_key)
        params = {"page": 1, "size": 20}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/regelingen/_zoek",
                json=body,
                headers=headers,
                params=params,
            )
            if response.status_code == 404:
                return []
            response.raise_for_status()
            data = response.json()

        regelingen = data.get("_embedded", {}).get("regelingen", [])
        return regelingen

    async def get_regeling(self, identificatie: str) -> Optional[dict]:
        """Get details of a specific regeling by its AKN identificatie."""
        uri_id = to_uri_identificatie(identificatie)
        headers = _dso_headers(self.api_key)
        headers.pop("Content-Crs", None)
        headers.pop("Content-Type", None)

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{self.base_url}/regelingen/{uri_id}",
                headers=headers,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()

    async def zoek_regeltekstannotaties(
        self,
        identificatie: str,
        x_rd: float,
        y_rd: float,
        page: int = 1,
        size: int = 50,
    ) -> list:
        """
        Get regeltekst annotations for a regeling filtered by location.
        These contain the actual rule texts (inhoud in XHTML).
        """
        uri_id = to_uri_identificatie(identificatie)
        body = {"geometrie": _point_geometry(x_rd, y_rd)}
        headers = _dso_headers(self.api_key)
        params = {"page": page, "size": size}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/regelingen/{uri_id}/regeltekstannotaties/_zoek",
                json=body,
                headers=headers,
                params=params,
            )
            if response.status_code in (404, 422):
                return []
            response.raise_for_status()
            data = response.json()

        embedded = data.get("_embedded", {})
        return embedded.get("regeltekstAnnotaties", [])

    async def zoek_divisieannotaties(
        self,
        identificatie: str,
        x_rd: float,
        y_rd: float,
        page: int = 1,
        size: int = 50,
    ) -> list:
        """
        Get divisie annotations for a regeling filtered by location.
        Used for regelingen with free-text structure (e.g. omgevingsvisies).
        """
        uri_id = to_uri_identificatie(identificatie)
        body = {"geometrie": _point_geometry(x_rd, y_rd)}
        headers = _dso_headers(self.api_key)
        params = {"page": page, "size": size}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/regelingen/{uri_id}/divisieannotaties/_zoek",
                json=body,
                headers=headers,
                params=params,
            )
            if response.status_code in (404, 422):
                return []
            response.raise_for_status()
            data = response.json()

        embedded = data.get("_embedded", {})
        return embedded.get("divisieAnnotaties", [])

    async def get_regeling_documentstructuur(self, identificatie: str) -> Optional[dict]:
        """Get the document structure (table of contents) of a regeling."""
        uri_id = to_uri_identificatie(identificatie)
        headers = _dso_headers(self.api_key)
        headers.pop("Content-Crs", None)
        headers.pop("Content-Type", None)

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{self.base_url}/regelingen/{uri_id}/documentstructuur",
                headers=headers,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()


def extract_rules_text(regeltekst_annotaties: list[dict]) -> list[dict]:
    """
    Extract relevant text from regeltekstAnnotaties response.
    Returns list of {nummer, opschrift, tekst, activiteiten, gebiedsaanwijzingen}
    """
    rules = []
    for item in regeltekst_annotaties:
        regeltekst = item.get("regeltekst", {})
        nummer = regeltekst.get("nummer", "")
        opschrift = regeltekst.get("opschrift", "")
        inhoud_raw = regeltekst.get("inhoud", "")
        tekst = clean_xhtml(inhoud_raw) if inhoud_raw else ""

        activiteiten = []
        for act in item.get("activiteiten", []):
            act_naam = act.get("naam", "")
            if act_naam:
                activiteiten.append(act_naam)

        gebiedsaanwijzingen = []
        for ga in item.get("gebiedsaanwijzingen", []):
            ga_naam = ga.get("naam", "")
            if ga_naam:
                gebiedsaanwijzingen.append(ga_naam)

        if tekst or opschrift:
            rules.append({
                "nummer": nummer,
                "opschrift": opschrift,
                "tekst": tekst,
                "activiteiten": activiteiten,
                "gebiedsaanwijzingen": gebiedsaanwijzingen,
            })

    return rules


def extract_divisie_text(divisie_annotaties: list[dict]) -> list[dict]:
    """
    Extract relevant text from divisieAnnotaties response.
    """
    items = []
    for item in divisie_annotaties:
        divisie = item.get("divisie", item.get("divisietekst", {}))
        opschrift = divisie.get("opschrift", "")
        inhoud_raw = divisie.get("inhoud", "")
        tekst = clean_xhtml(inhoud_raw) if inhoud_raw else ""

        if tekst or opschrift:
            items.append({
                "opschrift": opschrift,
                "tekst": tekst,
            })
    return items


def format_rules_for_llm(rules: list[dict], divisies: list[dict] = None) -> str:
    """
    Format extracted rules into a clean text block for LLM input.
    """
    parts = []

    if rules:
        parts.append("=== REGELS UIT HET OMGEVINGSPLAN ===\n")
        for rule in rules:
            section = []
            if rule["nummer"] or rule["opschrift"]:
                header = " ".join(filter(None, [rule["nummer"], rule["opschrift"]]))
                section.append(f"**{header}**")
            if rule["activiteiten"]:
                section.append(f"Activiteiten: {', '.join(rule['activiteiten'])}")
            if rule["gebiedsaanwijzingen"]:
                section.append(f"Gebiedsaanwijzingen: {', '.join(rule['gebiedsaanwijzingen'])}")
            if rule["tekst"]:
                section.append(rule["tekst"])
            if section:
                parts.append("\n".join(section))
                parts.append("---")

    if divisies:
        parts.append("\n=== BELEIDSTEKSTEN ===\n")
        for div in divisies:
            section = []
            if div["opschrift"]:
                section.append(f"**{div['opschrift']}**")
            if div["tekst"]:
                section.append(div["tekst"])
            if section:
                parts.append("\n".join(section))
                parts.append("---")

    return "\n".join(parts)
