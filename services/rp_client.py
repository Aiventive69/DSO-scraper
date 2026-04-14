"""
Ruimtelijke Plannen V4 API client.
Used for querying legacy bestemmingsplannen that are still legally valid
during the Omgevingswet transition period.
"""

import httpx
from bs4 import BeautifulSoup
from typing import Optional, List

RP_BASE_URL = "https://ruimte.omgevingswet.overheid.nl/ruimtelijke-plannen/api/opvragen/v4"


class RPClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "X-Api-Key": api_key,
            "Accept": "application/json",
        }

    async def zoek_plannen(
        self,
        x_rd: float,
        y_rd: float,
        plan_types: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 10,
    ) -> list[dict]:
        """
        Search for ruimtelijke plannen (bestemmingsplannen) at a given RD New location.
        """
        if plan_types is None:
            plan_types = ["bestemmingsplan", "uitwerkingsplan", "wijzigingsplan"]

        params = {
            "geometrie.contains": f"POINT({x_rd} {y_rd})",
            "_pageSize": page_size,
            "_page": page,
        }

        all_plannen = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for plan_type in plan_types:
                p = {**params, "planType": plan_type}
                response = await client.get(
                    f"{RP_BASE_URL}/plannen",
                    headers=self.headers,
                    params=p,
                )
                if response.status_code in (404, 422):
                    continue
                if response.status_code != 200:
                    continue
                data = response.json()
                plannen = data.get("_embedded", {}).get("plannen", [])
                all_plannen.extend(plannen)

        all_plannen.sort(key=lambda p: p.get("datum", ""), reverse=True)
        return all_plannen

    async def get_bestemmingen(
        self,
        plan_id: str,
        x_rd: float,
        y_rd: float,
        page_size: int = 20,
    ) -> list[dict]:
        """
        Get bestemmingen (zoning designations) for a plan at a specific location.
        """
        params = {
            "geometrie.intersects": f"POINT({x_rd} {y_rd})",
            "_pageSize": page_size,
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{RP_BASE_URL}/plannen/{plan_id}/enkelbestemmingen",
                headers=self.headers,
                params=params,
            )
            if response.status_code in (404, 422):
                return []
            if response.status_code != 200:
                return []
            data = response.json()

        return data.get("_embedded", {}).get("enkelbestemmingen", [])

    async def get_dubbelbestemmingen(
        self,
        plan_id: str,
        x_rd: float,
        y_rd: float,
        page_size: int = 20,
    ) -> list[dict]:
        """Get dubbelbestemmingen for a plan at a specific location."""
        params = {
            "geometrie.intersects": f"POINT({x_rd} {y_rd})",
            "_pageSize": page_size,
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{RP_BASE_URL}/plannen/{plan_id}/dubbelbestemmingen",
                headers=self.headers,
                params=params,
            )
            if response.status_code in (404, 422):
                return []
            if response.status_code != 200:
                return []
            data = response.json()

        return data.get("_embedded", {}).get("dubbelbestemmingen", [])

    async def get_gebiedsaanduidingen(
        self,
        plan_id: str,
        x_rd: float,
        y_rd: float,
        page_size: int = 20,
    ) -> list:
        """Get gebiedsaanduidingen for a plan at a specific location."""
        params = {
            "geometrie.intersects": f"POINT({x_rd} {y_rd})",
            "_pageSize": page_size,
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{RP_BASE_URL}/plannen/{plan_id}/gebiedsaanduidingen",
                headers=self.headers,
                params=params,
            )
            if response.status_code in (404, 422):
                return []
            if response.status_code != 200:
                return []
            data = response.json()

        return data.get("_embedded", {}).get("gebiedsaanduidingen", [])

    async def get_teksten(
        self,
        plan_id: str,
        tekst_type: str = "regels",
        page_size: int = 100,
    ) -> list:
        """
        Get plan teksten (rules text) from the bestemmingsplan.
        Returns a list of tekst objects with titel and inhoud.
        """
        params = {"_pageSize": page_size}
        if tekst_type:
            params["tekstType"] = tekst_type

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{RP_BASE_URL}/plannen/{plan_id}/teksten",
                headers=self.headers,
                params=params,
            )
            if response.status_code in (404, 422):
                return []
            if response.status_code != 200:
                return []
            data = response.json()

        return data.get("_embedded", {}).get("teksten", [])

    def format_bestemmingsplan_for_llm(
        self,
        plan: dict,
        bestemmingen: list,
        dubbelbestemmingen: list,
        gebiedsaanduidingen: list,
        teksten: list = None,
    ) -> str:
        """Format bestemmingsplan data for LLM input."""
        parts = []
        plan_name = plan.get("naam", plan.get("id", "Bestemmingsplan"))
        plan_datum = plan.get("datum", "")
        parts.append(f"=== BESTEMMINGSPLAN: {plan_name} ({plan_datum}) ===\n")

        if bestemmingen:
            parts.append("ENKELBESTEMMINGEN:")
            for b in bestemmingen:
                naam = b.get("naam", "")
                artikelnummer = b.get("artikelnummer", "")
                beschrijving = b.get("beschrijving", "")
                if naam:
                    line = f"- {naam}"
                    if artikelnummer:
                        line += f" (artikel {artikelnummer})"
                    parts.append(line)
                if beschrijving:
                    bs = BeautifulSoup(beschrijving, "lxml")
                    parts.append(f"  {bs.get_text(separator=' ', strip=True)[:500]}")

        if dubbelbestemmingen:
            parts.append("\nDUBBELBESTEMMINGEN:")
            for d in dubbelbestemmingen:
                naam = d.get("naam", "")
                if naam:
                    parts.append(f"- {naam}")

        if gebiedsaanduidingen:
            parts.append("\nGEBIEDSAANDUIDINGEN:")
            for g in gebiedsaanduidingen:
                naam = g.get("naam", "")
                if naam:
                    parts.append(f"- {naam}")

        if teksten:
            parts.append("\nPLANREGELS (TEKSTEN):")
            for t in teksten:
                titel = t.get("titel", "")
                inhoud = t.get("inhoud", "") or ""
                if inhoud:
                    bs = BeautifulSoup(inhoud, "lxml")
                    clean = bs.get_text(separator=" ", strip=True)
                    if titel:
                        parts.append(f"\n{titel}")
                    parts.append(clean[:1000])

        return "\n".join(parts)
