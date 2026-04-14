# DSO Omgevingsplan Assistent

Een web-applicatie voor makelaars en vastgoedprofessionals om snel actuele omgevingsplan- en bestemmingsplaninformatie op te vragen voor een Nederlands adres.

## Wat doet het?

1. **Adres invoeren** – met autocomplete via PDOK Locatieserver
2. **Vraag stellen** – bijv. "Wat is de bestemming?" of "Zijn er bouwmogelijkheden?"
3. **Resultaat** – AI-samenvatting van de relevante regels + bronvermelding

### Stroom

```
Adres → PDOK Locatieserver → RD28992 coördinaten
     → DSO Presenteren v8  → Omgevingsplan + regels voor die locatie
     → Ruimtelijke Plannen V4 → Bestemmingsplan (fallback/aanvulling)
     → OpenAI GPT-4o       → Beantwoord de vraag op basis van de planteksten
```

## Installatie

### Vereisten

- Python 3.9+
- API-sleutels (zie hieronder)

### Starten

```bash
# 1. Kopieer de .env configuratie
cp .env.example .env

# 2. Voeg uw OpenAI sleutel toe in .env
nano .env   # of open met uw editor

# 3. Start de applicatie
./start.sh
```

De app is daarna bereikbaar op: **http://localhost:8000**

## Configuratie (.env)

| Variabele | Omschrijving | Standaard |
|-----------|-------------|-----------|
| `DSO_API_KEY` | API-sleutel DSO (omgevingsdocumenten) | pre-productie sleutel |
| `RP_API_KEY` | API-sleutel Ruimtelijke Plannen V4 | meegeleverd |
| `OPENAI_API_KEY` | OpenAI API-sleutel voor AI-samenvatting | _(leeg = geen AI)_ |
| `OPENAI_MODEL` | OpenAI model | `gpt-4o` |
| `DSO_PRODUCTION` | Gebruik productieomgeving DSO | `false` |

> **Let op:** De meegeleverde DSO-sleutel is voor de **pre-productieomgeving**. Deze omgeving heeft beperkte testdata. Voor volledige functionaliteit vraagt u een productie API-sleutel aan via het [Ontwikkelaarsportaal](https://developer.omgevingswet.overheid.nl/).

## API's

### DSO Omgevingsdocumenten Presenteren v8
- Pre-productie: `https://service.pre.omgevingswet.overheid.nl/publiek/omgevingsdocumenten/api/presenteren/v8/`
- Productie: `https://service.omgevingswet.overheid.nl/publiek/omgevingsdocumenten/api/presenteren/v8/`
- Coördinatenstelsel: **RD New (EPSG:28992)**

### Ruimtelijke Plannen V4
- Productie: `https://ruimte.omgevingswet.overheid.nl/ruimtelijke-plannen/api/opvragen/v4/`
- Gebruikt voor: bestemmingsplannen, TAM-omgevingsplannen

### PDOK Locatieserver
- URL: `https://api.pdok.nl/bzk/locatieserver/search/v3_1/`
- Gratis geocodeer-service voor Nederlandse adressen
- Geen API-sleutel vereist

## Projectstructuur

```
DSO LLM/
├── main.py                 # FastAPI backend
├── services/
│   ├── geocoder.py         # PDOK adresopzoek
│   ├── dso_client.py       # DSO Presenteren API v8
│   ├── rp_client.py        # Ruimtelijke Plannen V4 API
│   └── summarizer.py       # OpenAI samenvatting
├── static/
│   └── index.html          # Frontend (single-page app)
├── requirements.txt
├── .env.example
└── start.sh
```

## Snelle vragen (ingebouwd)

- Bestemming van het perceel
- Bouwmogelijkheden en maximale bouwhoogte
- Gebruiksregels en verboden activiteiten
- Regels voor uitbreiden / verbouwen
- Parkeer- en verkeersregels
- Volledig overzicht van alle regels

## Technische details

- **Backend**: FastAPI (Python)
- **Coördinaten**: PDOK geeft RD New (X,Y) terug → direct bruikbaar voor DSO API
- **CRS header**: `Content-Crs: http://www.opengis.net/def/crs/EPSG/0/28992`
- **ID-format DSO**: `/akn/nl/act/gm0363/...` → `_akn_nl_act_gm0363_...` (slashes en koppeltekens → underscores)

## Disclaimer

De informatie in deze applicatie is afkomstig van het Digitaal Stelsel Omgevingswet (DSO) en Ruimtelijkeplannen.nl. Raadpleeg altijd de officiële brondocumenten en eventueel een omgevingsjurist voor juridisch bindende informatie.
