"""
AI summarization service using OpenAI.
Summarizes DSO omgevingsplan content based on user questions.
"""

from openai import AsyncOpenAI
from typing import Optional

SYSTEM_PROMPT = """Je bent een expert assistent voor Nederlandse makelaars en vastgoedprofessionals.
Je analyseert bestemmingsplan- en omgevingsplanteksten en beantwoordt daar gerichte vragen over.

Instructies:
- Lees de volledige plantekst die je krijgt aangeleverd
- Beantwoord de gestelde vraag op basis van de plantekst
- Haal uitsluitend informatie uit de aangeleverde tekst; verzin niets
- Citeer het relevante artikelnummer of de sectie (bijv. "Artikel 4.1")
- Schrijf beknopt en praktisch: de lezer is een makelaar, geen jurist
- Gebruik bullet points als er meerdere relevante punten zijn
- Geen inleiding, geen herhaling van de vraag — ga direct naar het antwoord

Als de vraag gaat over de bestemming van een perceel, geef dan ALTIJD:
1. De naam van de bestemming en het artikelnummer
2. Een beknopte samenvatting van wat die bestemming toestaat (bestemmingsomschrijving)
3. De belangrijkste bouwregels (maximale hoogte, bouwvlak, etc.) indien vermeld
4. Eventuele bijzondere aanduidingen of gebruiksregels die relevant zijn

Sluit altijd af met één zin disclaimer over juridische zekerheid."""


async def summarize_with_openai(
    plan_text: str,
    vraag: str,
    adres: str,
    model: str,
    api_key: str,
    max_context_chars: int = 80000,
) -> str:
    """
    Answer the user's question based on the full plan text using OpenAI.
    """
    client = AsyncOpenAI(api_key=api_key)

    truncated_text = plan_text[:max_context_chars]
    if len(plan_text) > max_context_chars:
        truncated_text += f"\n\n[... tekst ingekort, {len(plan_text) - max_context_chars} tekens weggelaten ...]"

    user_message = f"""Adres: {adres}

Vraag: {vraag}

--- PLANTEKST ---
{truncated_text}
--- EINDE PLANTEKST ---

Beantwoord de vraag op basis van bovenstaande plantekst."""

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1500,
    )

    return response.choices[0].message.content or "Geen antwoord ontvangen."


def format_without_ai(
    plan_text: str,
    vraag: str,
    adres: str,
    max_chars: int = 5000,
) -> str:
    """
    Format plan information without AI summarization.
    Used as fallback when no OpenAI key is configured.
    """
    truncated = plan_text[:max_chars]
    if len(plan_text) > max_chars:
        truncated += f"\n\n[... tekst ingekort om weergave te beperken ...]"

    return f"""**Let op:** Geen OpenAI API-sleutel geconfigureerd. Hieronder de ruwe planteksten voor het adres **{adres}**.

Uw vraag: *{vraag}*

---

{truncated}

---
*Voor een AI-samenvatting, voeg uw OpenAI API-sleutel toe aan het .env bestand.*"""
