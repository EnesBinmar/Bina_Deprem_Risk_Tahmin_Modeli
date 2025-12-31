"""
Deprem Risk Tahmin Sistemi - Web Arayüzü
========================================
Flask tabanlı basit web arayüzü ile deprem risk tahmini
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)

# =============================================================================
# MODEL VE VERİLERİ YÜKLE
# =============================================================================

# Fay hattı koordinatları
fay_hatti = [
    (37.0500, 36.4500), (36.9000, 36.4000), (36.8000, 36.3500),
    (36.7000, 36.3000), (36.6000, 36.2500), (36.5000, 36.2000),
    (36.4000, 36.1500), (36.3000, 36.1000), (36.2000, 36.0500),
    (36.1000, 36.0000), (36.0000, 35.9500),
]

# İlçe koordinatları
ilce_koordinatlari = {
    'İskenderun': (36.5817, 36.1700),
    'Antakya': (36.2025, 36.1597),
    'Arsuz': (36.4139, 35.8875),
    'Defne': (36.2300, 36.1400),
    'Samandağ': (36.0833, 35.9667),
    'Dörtyol': (36.8500, 36.2167),
    'Belen': (36.4917, 36.1917),
    'Kırıkhan': (36.5000, 36.3667),
    'Payas': (36.7583, 36.2250),
    'Erzin': (36.9500, 36.2000),
    'Hassa': (36.8000, 36.5167),
    'Altınözü': (36.1167, 36.2500),
    'Reyhanlı': (36.2667, 36.5667),
    'Yayladağı': (35.9000, 36.0500),
}

# Mahalle koordinatları
mahalle_koordinatlari = {
    'İsmet İnönü': (36.5850, 36.1750),
    'Sakarya': (36.5780, 36.1680),
    'Numune': (36.5900, 36.1800),
    'Mustafa Kemal': (36.5750, 36.1650),
    'Dumlupınar': (36.5700, 36.1600),
    'Denizciler': (36.5850, 36.1720),
    'Yunus Emre': (36.5800, 36.1700),
    'Kurtuluş': (36.5820, 36.1680),
    'Pirireis': (36.5750, 36.1750),
    'Cumhuriyet': (36.5830, 36.1650),
    'Modernevler': (36.5880, 36.1780),
    'Karaağaç': (36.4200, 35.9000),
    'Saraykent': (36.2100, 36.1650),
    'Akevler': (36.2000, 36.1550),
    'Mızraklı': (36.1000, 35.9800),
    'Fatih': (36.4950, 36.1950),
}

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def nokta_dogru_mesafe(nokta, p1, p2):
    px, py = nokta
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return haversine(px, py, x1, y1)
    t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)))
    return haversine(px, py, x1 + t*dx, y1 + t*dy)

def fay_uzakligi_hesapla(lat, lon):
    return min(nokta_dogru_mesafe((lat, lon), fay_hatti[i], fay_hatti[i+1]) 
               for i in range(len(fay_hatti)-1))

def koordinat_bul(ilce, mahalle=None):
    if mahalle:
        mah_temiz = mahalle.strip().lower()
        for mah, koord in mahalle_koordinatlari.items():
            if mah.lower() in mah_temiz or mah_temiz in mah.lower():
                return koord
    return ilce_koordinatlari.get(ilce, (36.5817, 36.1700))

def risk_tahmin(bina_yasi, kat_sayisi, fay_uzakligi_km, yumusak_kat, 
                kapali_cikma, nizami, simetrik):
    """Risk tahmini yap"""
    
    # Bina yaşı dönüşümü
    bina_yasi_map = {'0-5': 2.5, '6-10': 8, '11-20': 15, '21-30': 25, '30+': 40}
    yas_num = bina_yasi_map.get(bina_yasi, 15)
    
    # Binary dönüşümler
    yumusak_num = 1 if yumusak_kat else 0
    kapali_num = 1 if kapali_cikma else 0
    nizami_num = 1 if nizami else 0
    simetrik_num = 1 if simetrik else 0
    
    # Fay özellikleri
    fay_yakinligi = 25 - min(fay_uzakligi_km, 25)
    fay_risk = fay_yakinligi * 2
    
    # Risk skoru hesapla
    risk_skoru = (
        yas_num * 0.4 +
        yumusak_num * 20 +
        (1 - nizami_num) * 15 +
        (1 - simetrik_num) * 12 +
        kapali_num * 8 +
        kat_sayisi * 2 +
        fay_yakinligi * 1.5
    )
    
    # Risk kategorisi belirle
    if risk_skoru < 30:
        kategori = "DÜŞÜK RİSK"
        renk = "success"
        emoji = "✅"
        aciklama = "Binanız deprem açısından güvenli görünüyor."
    elif risk_skoru < 55:
        kategori = "ORTA RİSK"
        renk = "warning"
        emoji = "⚠️"
        aciklama = "Binanızda bazı risk faktörleri mevcut. Güçlendirme değerlendirmesi yapılabilir."
    else:
        kategori = "YÜKSEK RİSK"
        renk = "danger"
        emoji = "🔴"
        aciklama = "Binanız yüksek risk altında! Acil olarak uzman değerlendirmesi önerilir."
    
    # Detaylı analiz
    risk_faktorleri = []
    
    if yas_num >= 25:
        risk_faktorleri.append(("Bina Yaşı", "Bina 25 yıldan eski, yapısal zayıflama riski var", "high"))
    elif yas_num >= 15:
        risk_faktorleri.append(("Bina Yaşı", "Bina orta yaşlı, düzenli kontrol önerilir", "medium"))
    else:
        risk_faktorleri.append(("Bina Yaşı", "Bina nispeten yeni", "low"))
    
    if fay_uzakligi_km < 5:
        risk_faktorleri.append(("Fay Uzaklığı", f"Fay hattına çok yakın ({fay_uzakligi_km:.1f} km)", "high"))
    elif fay_uzakligi_km < 10:
        risk_faktorleri.append(("Fay Uzaklığı", f"Fay hattına yakın ({fay_uzakligi_km:.1f} km)", "medium"))
    else:
        risk_faktorleri.append(("Fay Uzaklığı", f"Fay hattına uzak ({fay_uzakligi_km:.1f} km)", "low"))
    
    if yumusak_num:
        risk_faktorleri.append(("Yumuşak Kat", "Zemin katta yumuşak kat var - yüksek risk!", "high"))
    
    if not nizami_num:
        risk_faktorleri.append(("Yapı Düzeni", "Bina nizami düzende değil", "medium"))
    
    if not simetrik_num:
        risk_faktorleri.append(("Yapı Simetrisi", "Bina asimetrik yapıda", "medium"))
    
    if kapali_num:
        risk_faktorleri.append(("Kapalı Çıkma", "Binada kapalı çıkma var", "medium"))
    
    if kat_sayisi >= 8:
        risk_faktorleri.append(("Kat Sayısı", f"Yüksek katlı bina ({kat_sayisi} kat)", "medium"))
    
    return {
        'risk_skoru': round(risk_skoru, 1),
        'kategori': kategori,
        'renk': renk,
        'emoji': emoji,
        'aciklama': aciklama,
        'fay_uzakligi': round(fay_uzakligi_km, 2),
        'risk_faktorleri': risk_faktorleri
    }

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html', ilceler=list(ilce_koordinatlari.keys()))

@app.route('/tahmin', methods=['POST'])
def tahmin():
    try:
        # Form verilerini al
        ilce = request.form.get('ilce', 'İskenderun')
        mahalle = request.form.get('mahalle', '')
        bina_yasi = request.form.get('bina_yasi', '11-20')
        kat_sayisi = int(request.form.get('kat_sayisi', 5))
        yumusak_kat = request.form.get('yumusak_kat') == 'on'
        kapali_cikma = request.form.get('kapali_cikma') == 'on'
        nizami = request.form.get('nizami') == 'on'
        simetrik = request.form.get('simetrik') == 'on'
        
        # Fay uzaklığı hesapla
        lat, lon = koordinat_bul(ilce, mahalle)
        fay_uzakligi = fay_uzakligi_hesapla(lat, lon)
        
        # Risk tahmini yap
        sonuc = risk_tahmin(bina_yasi, kat_sayisi, fay_uzakligi, 
                           yumusak_kat, kapali_cikma, nizami, simetrik)
        
        sonuc['ilce'] = ilce
        sonuc['mahalle'] = mahalle if mahalle else '-'
        sonuc['bina_yasi'] = bina_yasi
        sonuc['kat_sayisi'] = kat_sayisi
        
        return render_template('sonuc.html', sonuc=sonuc)
    
    except Exception as e:
        return render_template('error.html', hata=str(e))

@app.route('/api/tahmin', methods=['POST'])
def api_tahmin():
    """API endpoint for JSON requests"""
    try:
        data = request.json
        
        ilce = data.get('ilce', 'İskenderun')
        mahalle = data.get('mahalle', '')
        bina_yasi = data.get('bina_yasi', '11-20')
        kat_sayisi = int(data.get('kat_sayisi', 5))
        yumusak_kat = data.get('yumusak_kat', False)
        kapali_cikma = data.get('kapali_cikma', False)
        nizami = data.get('nizami', True)
        simetrik = data.get('simetrik', True)
        
        lat, lon = koordinat_bul(ilce, mahalle)
        fay_uzakligi = fay_uzakligi_hesapla(lat, lon)
        
        sonuc = risk_tahmin(bina_yasi, kat_sayisi, fay_uzakligi,
                           yumusak_kat, kapali_cikma, nizami, simetrik)
        
        return jsonify(sonuc)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("=" * 50)
    print("🏠 DEPREM RİSK TAHMİN SİSTEMİ")
    print("=" * 50)
    print("\n🌐 Web arayüzü başlatılıyor...")
    print("📍 Adres: http://localhost:5000")
    print("\nDurdurmak için Ctrl+C basın")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
