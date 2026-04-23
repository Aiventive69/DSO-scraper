"""
DSO Omgevingsplan Assistent - FastAPI backend
Geocodes Dutch addresses and queries the DSO for omgevingsplan data,
then summarizes it using OpenAI based on the user's question.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from services.geocoder import geocode_address, suggest_address
from services.dso_client import (
    DSOClient,
    extract_rules_text,
    extract_divisie_text,
    format_rules_for_llm,
)
from services.wms_client import get_bestemmingsplan_data, format_bestemmingsplan_for_llm
from services.summarizer import summarize_with_openai, format_without_ai

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DSO_API_KEY = os.getenv("DSO_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DSO_PRODUCTION = os.getenv("DSO_PRODUCTION", "").lower() in ("true", "1", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"DSO Omgevingsplan Assistent gestart")
    logger.info(f"DSO omgeving: {'productie' if DSO_PRODUCTION else 'pre-productie'}")
    logger.info(f"AI samenvatting: {'actief (OpenAI)' if OPENAI_API_KEY else 'niet geconfigureerd'}")
    yield


app = FastAPI(
    title="DSO Omgevingsplan Assistent",
    description="Bevraag het omgevingsplan van een gemeente op basis van een adres en vraag.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    adres: str
    vraag: str
    include_bestemmingsplan: bool = True


class QueryResponse(BaseModel):
    adres: str
    gemeente: str
    omgevingsplan_naam: Optional[str] = None
    bestemmingsplan_naam: Optional[str] = None
    bestemming: Optional[str] = None
    samenvatting: str
    heeft_omgevingsplan: bool
    heeft_bestemmingsplan: bool
    ai_gebruikt: bool
    bronnen: list
    geocode_waarschuwing: Optional[str] = None


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/api/suggest")
async def suggest(q: str = Query(..., min_length=2)):
    """Address autocomplete using PDOK Locatieserver."""
    try:
        suggestions = await suggest_address(q)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Suggest error: {e}")
        raise HTTPException(status_code=500, detail="Autocomplete niet beschikbaar")


@app.post("/api/query", response_model=QueryResponse)
async def query_omgevingsplan(req: QueryRequest):
    """
    Main endpoint: geocode address, query DSO omgevingsplan, summarize with AI.
    """
    logger.info(f"Query voor adres: {req.adres}")

    # Step 1: Geocode address
    coords = await geocode_address(req.adres)
    if not coords:
        raise HTTPException(
            status_code=404,
            detail=f"Adres '{req.adres}' niet gevonden. Controleer het adres en probeer opnieuw.",
        )

    logger.info(
        f"Geocoded: {coords['adres_display']} (type={coords.get('geocode_type','?')}) "
        f"-> RD({coords['x_rd']}, {coords['y_rd']})"
    )

    # Warn when we only found a street, not an exact address
    geocode_waarschuwing = None
    if not coords.get("is_exact_adres", True):
        geocode_waarschuwing = (
            f"⚠️ Geen exact adres gevonden voor '{req.adres}'. "
            f"Resultaten zijn gebaseerd op de straat: **{coords['adres_display']}**. "
            f"Voeg een huisnummer toe voor perceel-specifieke informatie."
        )
        logger.warning(f"Straat-niveau geocoding: {coords['adres_display']}")

    all_plan_texts = []
    if geocode_waarschuwing:
        all_plan_texts.append(geocode_waarschuwing)
    bronnen = []
    omgevingsplan_naam = None
    bestemmingsplan_naam = None
    bestemming_naam = None
    heeft_omgevingsplan = False
    heeft_bestemmingsplan = False
    dso_heeft_inhoud = False

    # Step 2: Query DSO omgevingsplan (primaire bron — wordt rijker naarmate gemeenten migreren)
    if DSO_API_KEY:
        try:
            dso = DSOClient(api_key=DSO_API_KEY, use_production=DSO_PRODUCTION)
            regelingen = await dso.zoek_regelingen(
                x_rd=coords["x_rd"],
                y_rd=coords["y_rd"],
                gemeente_code=coords["gemeente_code"] if coords["gemeente_code"] else None,
                type_bevoegd_gezag=["gemeente"],
            )

            if not regelingen:
                logger.info("Geen regelingen met gemeente filter, probeer zonder filter...")
                regelingen = await dso.zoek_regelingen(
                    x_rd=coords["x_rd"],
                    y_rd=coords["y_rd"],
                    gemeente_code=None,
                    type_bevoegd_gezag=["gemeente"],
                )

            omgevingsplannen = [
                r for r in regelingen
                if "omgevingsplan" in r.get("type", {}).get("waarde", "").lower()
            ]

            for regeling in omgevingsplannen[:2]:
                regeling_id = regeling.get("identificatie", "")
                if not regeling_id:
                    continue
                opschrift = regeling.get("opschrift", regeling_id)
                heeft_omgevingsplan = True
                if not omgevingsplan_naam:
                    omgevingsplan_naam = opschrift

                # Probeer OW-geannoteerde regelteksten op te halen
                regelteksten = await dso.zoek_regeltekstannotaties(
                    identificatie=regeling_id,
                    x_rd=coords["x_rd"],
                    y_rd=coords["y_rd"],
                )
                divisies = await dso.zoek_divisieannotaties(
                    identificatie=regeling_id,
                    x_rd=coords["x_rd"],
                    y_rd=coords["y_rd"],
                )

                rules = extract_rules_text(regelteksten)
                div_items = extract_divisie_text(divisies)

                logger.info(f"DSO '{opschrift}': {len(rules)} regels, {len(div_items)} divisies")

                if rules or div_items:
                    dso_heeft_inhoud = True
                    plan_text = format_rules_for_llm(rules, div_items)
                    all_plan_texts.append(f"### Omgevingsplan: {opschrift}\n\n{plan_text}")
                    bronnen.append({
                        "type": "omgevingsplan",
                        "naam": opschrift,
                        "identificatie": regeling_id,
                        "bron": "DSO Omgevingsdocumenten v8 (OW-geannoteerd)",
                    })
                else:
                    # DSO heeft het plan maar nog geen OW-inhoud
                    bronnen.append({
                        "type": "omgevingsplan",
                        "naam": opschrift,
                        "identificatie": regeling_id,
                        "bron": "DSO Omgevingsdocumenten v8",
                    })

        except Exception as e:
            logger.error(f"DSO query error: {e}", exc_info=True)

    # Step 3: Fallback naar PDOK WMS + ruimtelijkeplannen.nl als DSO geen inhoud heeft
    if not dso_heeft_inhoud:
        logger.info("DSO heeft geen OW-inhoud — fallback naar PDOK WMS / ruimtelijkeplannen.nl")
        try:
            bp_data = await get_bestemmingsplan_data(
                x_rd=coords["x_rd"],
                y_rd=coords["y_rd"],
            )

            if bp_data.get("plan_naam"):
                heeft_bestemmingsplan = True
                bestemmingsplan_naam = bp_data["plan_naam"]
                bestemming_naam = bp_data.get("bestemming")

                logger.info(
                    f"Bestemmingsplan: {bestemmingsplan_naam} | "
                    f"Bestemming: {bestemming_naam} | "
                    f"Bouwvlak: {bp_data.get('bouwvlak')} | "
                    f"Maatvoering: {bp_data.get('maatvoering')}"
                )

                bron_entry = {
                    "type": "bestemmingsplan",
                    "naam": bestemmingsplan_naam,
                    "bron": "PDOK Ruimtelijke Plannen WMS + ruimtelijkeplannen.nl",
                }
                if bp_data.get("plan_id"):
                    bron_entry["identificatie"] = bp_data["plan_id"]
                if bp_data.get("plan_datum"):
                    bron_entry["datum"] = bp_data["plan_datum"]
                if bp_data.get("bestemming_url"):
                    bron_entry["url"] = bp_data["bestemming_url"]
                bronnen.append(bron_entry)

                bp_text = format_bestemmingsplan_for_llm(bp_data)
                if bp_text.strip():
                    if heeft_omgevingsplan and omgevingsplan_naam:
                        # Voeg context toe dat het bestemmingsplan geldt als omgevingsplan van rechtswege
                        all_plan_texts.append(
                            f"_Het omgevingsplan '{omgevingsplan_naam}' is van kracht, maar de "
                            f"OW-geannoteerde regels zijn nog niet gepubliceerd. "
                            f"Onderstaande bestemmingsplanregels zijn de geldende inhoudelijke regels._\n\n"
                            + bp_text
                        )
                    else:
                        all_plan_texts.append(bp_text)

        except Exception as e:
            logger.error(f"WMS/bestemmingsplan query error: {e}", exc_info=True)

    # Step 4: Compose combined text and summarize
    if not all_plan_texts:
        combined_text = (
            "Geen bestemmingsplan- of omgevingsplaninformatie gevonden voor dit adres. "
            "Het adres valt mogelijk buiten de beschikbare plangebieden in de PDOK kaartdienst."
        )
        samenvatting = combined_text
        ai_gebruikt = False
    else:
        combined_text = "\n\n".join(all_plan_texts)

        if OPENAI_API_KEY:
            try:
                samenvatting = await summarize_with_openai(
                    plan_text=combined_text,
                    vraag=req.vraag,
                    adres=coords["adres_display"],
                    model=OPENAI_MODEL,
                    api_key=OPENAI_API_KEY,
                )
                ai_gebruikt = True
            except Exception as e:
                logger.error(f"OpenAI error: {e}", exc_info=True)
                samenvatting = format_without_ai(combined_text, req.vraag, coords["adres_display"])
                ai_gebruikt = False
        else:
            samenvatting = format_without_ai(combined_text, req.vraag, coords["adres_display"])
            ai_gebruikt = False

    return QueryResponse(
        adres=coords["adres_display"],
        gemeente=coords["gemeente"],
        omgevingsplan_naam=omgevingsplan_naam,
        bestemmingsplan_naam=bestemmingsplan_naam,
        bestemming=bestemming_naam,
        samenvatting=samenvatting,
        heeft_omgevingsplan=heeft_omgevingsplan,
        heeft_bestemmingsplan=heeft_bestemmingsplan,
        ai_gebruikt=ai_gebruikt,
        bronnen=bronnen,
        geocode_waarschuwing=geocode_waarschuwing,
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "dso_omgeving": "productie" if DSO_PRODUCTION else "pre-productie",
        "ai_actief": bool(OPENAI_API_KEY),
        "ai_model": OPENAI_MODEL if OPENAI_API_KEY else None,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
