"""
AI summarization service using OpenAI.
Summarizes DSO omgevingsplan content based on user questions.

Strategy for large documents:
- Split the full plan text into overlapping chunks of CHUNK_SIZE chars
- Query all chunks in parallel (each chunk = one GPT call)
- Each call extracts relevant info from its chunk
- Final GPT call synthesizes all partial results into one answer
"""

import asyncio
import logging
from openai import AsyncOpenAI, RateLimitError
from typing import Optional, List

logger = logging.getLogger(__name__)

CHUNK_SIZE = 80_000    # chars per chunk sent to GPT
CHUNK_OVERLAP = 1_000  # overlap between chunks to avoid cutting mid-sentence
MAX_CONCURRENT = 3     # max parallel OpenAI calls (prevents TPM rate limit burst)

SYSTEM_PROMPT = """Je bent een expert assistent voor Nederlandse makelaars en vastgoedprofessionals.
Je analyseert bestemmingsplan- en omgevingsplanteksten en beantwoordt daar gerichte vragen over.

Instructies:
- Lees de volledige plantekst die je krijgt aangeleverd
- Beantwoord de gestelde vraag op basis van de plantekst
- Haal uitsluitend informatie uit de aangeleverde tekst; verzin niets
- Citeer het relevante artikelnummer of de sectie (bijv. "Artikel 4.1" of "lid 4.2.1")
- Schrijf beknopt en praktisch: de lezer is een makelaar, geen jurist
- Gebruik bullet points als er meerdere relevante punten zijn
- Geen inleiding, geen herhaling van de vraag — ga direct naar het antwoord

Als de vraag gaat over de bestemming van een perceel, geef dan ALTIJD:
1. De naam van de bestemming en het artikelnummer
2. Een beknopte samenvatting van wat die bestemming toestaat (bestemmingsomschrijving)
3. De belangrijkste bouwregels die in de tekst staan (hoogte, bouwvlak, bebouwingspercentage, etc.)
4. Eventuele bijzondere aanduidingen, nadere eisen of afwijkingsregels

Als er werkelijk geen informatie over het gevraagde onderwerp in de tekst staat,
zeg dan duidelijk: "De aangeleverde plantekst bevat geen regels over [onderwerp]."

Sluit altijd af met één zin disclaimer over juridische zekerheid."""

CHUNK_EXTRACT_PROMPT = """Je krijgt een DEEL van een bestemmingsplan of omgevingsplan.
Extraheer ALLEEN de tekst die relevant is voor de onderstaande vraag.
Kopieer de relevante artikelen, leden en zinnen letterlijk — voeg geen samenvatting toe.
Als er niets relevant staat in dit deel, antwoord dan alleen: "GEEN RELEVANTE INFO IN DIT DEEL."
Vermeld altijd het artikelnummer als dat aanwezig is."""

SYNTHESIZE_PROMPT = """Je hebt meerdere extracten ontvangen uit verschillende delen van een bestemmingsplan.
Combineer deze tot één volledig, samenhangend antwoord op de vraag.
- Verwijder dubbele informatie
- Citeer artikelnummers
- Schrijf beknopt en praktisch voor een makelaar
- Gebruik bullet points voor meerdere punten
- Geen inleiding, geen herhaling van de vraag — ga direct naar het antwoord
Sluit af met één zin disclaimer over juridische zekerheid."""


def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


async def _call_openai_with_retry(client: AsyncOpenAI, **kwargs) -> str:
    """Call OpenAI with automatic retry on 429 rate limit errors."""
    max_retries = 4
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Rate limit (429), wacht {delay:.0f}s en probeer opnieuw... (poging {attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)
            delay *= 2  # exponential backoff: 2s, 4s, 8s


async def _extract_from_chunk(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    vraag: str,
    adres: str,
    model: str,
) -> str:
    """Ask GPT to extract relevant content from a single chunk (rate-limited)."""
    user_message = (
        f"Adres: {adres}\n"
        f"Vraag: {vraag}\n\n"
        f"--- PLANTEKST (deel {chunk_index + 1} van {total_chunks}) ---\n"
        f"{chunk}\n"
        f"--- EINDE DEEL ---\n\n"
        f"Extraheer alle tekst uit dit deel die relevant is voor de vraag."
    )
    async with semaphore:
        result = await _call_openai_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": CHUNK_EXTRACT_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
    return result or "GEEN RELEVANTE INFO IN DIT DEEL."


async def _synthesize(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    extracts: List[str],
    vraag: str,
    adres: str,
    model: str,
) -> str:
    """Combine partial extracts into one final answer."""
    relevant = [e for e in extracts if "GEEN RELEVANTE INFO IN DIT DEEL" not in e]

    if not relevant:
        return (
            "De aangeleverde plantekst bevat geen specifieke informatie over dit onderwerp. "
            "Raadpleeg de officiële plantekst of neem contact op met de gemeente voor zekerheid."
        )

    combined = "\n\n---\n\n".join(
        f"[Uittreksel {i + 1}]\n{e}" for i, e in enumerate(relevant)
    )

    user_message = (
        f"Adres: {adres}\n"
        f"Vraag: {vraag}\n\n"
        f"Hieronder de relevante uittreksels uit het bestemmingsplan:\n\n"
        f"{combined}\n\n"
        f"Geef nu één volledig antwoord op de vraag."
    )

    async with semaphore:
        result = await _call_openai_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": SYNTHESIZE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=2500,
        )
    return result or "Geen antwoord ontvangen."


async def summarize_with_openai(
    plan_text: str,
    vraag: str,
    adres: str,
    model: str,
    api_key: str,
    max_context_chars: int = 80000,  # kept for API compatibility, no longer used as hard limit
) -> str:
    """
    Answer the user's question based on the full plan text using OpenAI.

    For texts longer than CHUNK_SIZE:
    - Split into overlapping chunks
    - Extract relevant content from ALL chunks in parallel
    - Synthesize into one final answer

    For texts shorter than CHUNK_SIZE:
    - Single GPT call (fast path)
    """
    client = AsyncOpenAI(api_key=api_key)
    chunks = _split_into_chunks(plan_text)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    if len(chunks) == 1:
        # Fast path: fits in a single call
        user_message = (
            f"Adres: {adres}\n\n"
            f"Vraag: {vraag}\n\n"
            f"--- PLANTEKST ---\n{plan_text}\n--- EINDE PLANTEKST ---\n\n"
            f"Beantwoord de vraag op basis van bovenstaande plantekst."
        )
        async with semaphore:
            return await _call_openai_with_retry(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=2500,
            ) or "Geen antwoord ontvangen."

    # Multi-chunk path: max MAX_CONCURRENT parallel extractions + synthesis
    logger.info(f"Verwerking in {len(chunks)} chunks (max {MAX_CONCURRENT} parallel)")
    total = len(chunks)
    extract_tasks = [
        _extract_from_chunk(client, semaphore, chunk, i, total, vraag, adres, model)
        for i, chunk in enumerate(chunks)
    ]
    extracts = await asyncio.gather(*extract_tasks)
    return await _synthesize(client, semaphore, list(extracts), vraag, adres, model)


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
        truncated += "\n\n[... tekst ingekort om weergave te beperken ...]"

    return (
        f"**Let op:** Geen OpenAI API-sleutel geconfigureerd. "
        f"Hieronder de ruwe planteksten voor het adres **{adres}**.\n\n"
        f"Uw vraag: *{vraag}*\n\n---\n\n{truncated}\n\n---\n"
        f"*Voor een AI-samenvatting, voeg uw OpenAI API-sleutel toe aan het .env bestand.*"
    )
