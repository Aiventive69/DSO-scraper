#!/bin/bash
# DSO Omgevingsplan Assistent - Start script
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🏛️  DSO Omgevingsplan Assistent"
echo "================================"

# Check if venv_packages exists, install if needed
if [ ! -d "venv_packages" ]; then
    echo "📦 Afhankelijkheden installeren..."
    python3 -m pip install -r requirements.txt --target=./venv_packages -q
    echo "✓ Klaar"
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  Geen .env bestand gevonden. Kopieer .env.example naar .env:"
    echo "   cp .env.example .env"
    echo "   # Voeg uw OpenAI API-sleutel toe voor AI-samenvattingen"
    echo ""
fi

echo ""
echo "🚀 Server starten op http://localhost:8000"
echo "   Druk op Ctrl+C om te stoppen"
echo ""

export PYTHONPATH="$SCRIPT_DIR/venv_packages:$PYTHONPATH"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
