#!/bin/bash
# ===========================================
# 🏠 Deprem Risk Tahmin Sistemi
# Tek Komutla Kurulum ve Başlatma
# ===========================================

echo "=========================================="
echo "🏠 DEPREM RİSK TAHMİN SİSTEMİ"
echo "=========================================="

cd "$(dirname "$0")"

# Sanal ortam yoksa oluştur
if [ ! -d ".venv" ]; then
    echo "📦 Sanal ortam oluşturuluyor..."
    python3 -m venv .venv
fi

# Aktif et
source .venv/bin/activate

# Paketleri yükle
echo "📥 Paketler yükleniyor..."
pip install -q flask pandas numpy scikit-learn 2>/dev/null

# Başlat
echo ""
echo "🚀 Başlatılıyor..."
echo "📍 Adres: http://localhost:5000"
echo "❌ Durdurmak için: Ctrl+C"
echo "=========================================="
python web_app.py
