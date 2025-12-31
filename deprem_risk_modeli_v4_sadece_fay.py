"""
Deprem Risk Skoru Tahmin Modeli v4.0 (SADECE FAY UZAKLIĞI)
==========================================================
Bu modelde ilçe/mahalle bilgisi doğrudan kullanılmaz.
Lokasyon bilgisi SADECE fay hattına uzaklık olarak modele girer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FAY UZAKLIĞI HESAPLAMA
# =============================================================================

# İlçe koordinatları (sadece fay uzaklığı hesabı için)
ilce_koordinatlari = {
    'İskenderun': (36.5817, 36.1700), 'Antakya': (36.2025, 36.1597),
    'Arsuz': (36.4139, 35.8875), 'Defne': (36.2300, 36.1400),
    'Samandağ': (36.0833, 35.9667), 'Dörtyol': (36.8500, 36.2167),
    'Belen': (36.4917, 36.1917), 'Kırıkhan': (36.5000, 36.3667),
}

mahalle_koordinatlari = {
    'İsmet İnönü': (36.5850, 36.1750), 'Sakarya': (36.5780, 36.1680),
    'Numune': (36.5900, 36.1800), 'Mustafa Kemal': (36.5750, 36.1650),
    'Dumlupınar': (36.5700, 36.1600), 'Denizciler': (36.5850, 36.1720),
    'Yunus Emre': (36.5800, 36.1700), 'Kurtuluş': (36.5820, 36.1680),
    'Pirireis': (36.5750, 36.1750), 'Cumhuriyet': (36.5830, 36.1650),
    'Modernevler': (36.5880, 36.1780), 'Meydan': (36.5817, 36.1700),
    'Nardüzü': (36.5600, 36.1500), 'Karaağaç': (36.4200, 35.9000),
    'Övündük': (36.4100, 35.8900), 'Öğündük': (36.4100, 35.8900),
    'Saraykent': (36.2100, 36.1650), 'Akevler': (36.2000, 36.1550),
    'Mızraklı': (36.1000, 35.9800), 'Gümüşgöze': (36.2350, 36.1450),
    'Fatih': (36.4950, 36.1950), 'Sarımazı': (36.4900, 36.1900),
}

# Doğu Anadolu Fay Hattı (DAFZ) - Hatay segmenti
fay_hatti = [
    (37.0500, 36.4500), (36.9000, 36.4000), (36.8000, 36.3500),
    (36.7000, 36.3000), (36.6000, 36.2500), (36.5000, 36.2000),
    (36.4000, 36.1500), (36.3000, 36.1000), (36.2000, 36.0500),
    (36.1000, 36.0000), (36.0000, 35.9500),
]

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
    return ilce_koordinatlari.get(ilce.strip() if ilce else 'İskenderun', (36.5817, 36.1700))

# =============================================================================
# ANA PROGRAM
# =============================================================================

print("=" * 70)
print("DEPREM RİSK TAHMİN MODELİ v4.0")
print("(İLÇE/MAHALLE YOK - SADECE FAY UZAKLIĞI)")
print("=" * 70)

# Veri yükleme
df = pd.read_csv('/home/enesbinmar/Masaüstü/deprem_risk_tahmin_modeli/data/Deprem Risk Skoru Veri - Form Yanıtları 1.csv')
df.columns = ['zaman_damgasi', 'deprem_hissetti_mi', 'binada_miydi', 'bina_yasi',
              'ilce', 'mahalle', 'kat_sayisi', 'hasar_durumu', 'yumusak_kat',
              'kapali_cikma', 'nizami_duzende_mi', 'simetrik_yapi_mi', 'asma_kat']

print(f"\n📊 Toplam kayıt: {len(df)}")

# =============================================================================
# FAY UZAKLIĞI HESAPLAMA
# =============================================================================
print("\n🌋 Fay hattına uzaklıklar hesaplanıyor...")

df['fay_uzakligi'] = df.apply(
    lambda row: fay_uzakligi_hesapla(*koordinat_bul(row['ilce'], row['mahalle'])), axis=1
)

print(f"✅ Fay uzaklığı hesaplandı!")
print(f"   Min: {df['fay_uzakligi'].min():.2f} km")
print(f"   Max: {df['fay_uzakligi'].max():.2f} km")
print(f"   Ort: {df['fay_uzakligi'].mean():.2f} km")

# =============================================================================
# ÖZELLİK MÜHENDİSLİĞİ (İLÇE/MAHALLE OLMADAN)
# =============================================================================
print("\n" + "=" * 70)
print("🔧 ÖZELLİK MÜHENDİSLİĞİ (İLÇE/MAHALLE YOK)")
print("=" * 70)

# Hasar gruplandırma
def hasar_grupla(hasar):
    if hasar == 'Hasarsız': return 'Düşük Risk'
    elif hasar == 'Az Hasarlı': return 'Orta Risk'
    else: return 'Yüksek Risk'

df['hasar_grubu'] = df['hasar_durumu'].apply(hasar_grupla)
hasar_mapping = {'Düşük Risk': 0, 'Orta Risk': 1, 'Yüksek Risk': 2}
df['hasar_skoru'] = df['hasar_grubu'].map(hasar_mapping)

# Bina özellikleri
bina_yasi_map = {'0-5': 2.5, '6-10': 8, '11-20': 15, '21-30': 25, '30+': 40}
df['bina_yasi_sayi'] = df['bina_yasi'].map(bina_yasi_map)

binary_map = {'Evet': 1, 'Hayır': 0}
df['yumusak_kat_enc'] = df['yumusak_kat'].map(binary_map)
df['kapali_cikma_enc'] = df['kapali_cikma'].map(binary_map)
df['nizami_enc'] = df['nizami_duzende_mi'].apply(lambda x: 1 if 'Evet' in str(x) else 0)
df['simetrik_enc'] = df['simetrik_yapi_mi'].apply(lambda x: 1 if 'Evet' in str(x) or 'kare' in str(x).lower() else 0)
df['kat_sayisi_num'] = pd.to_numeric(df['kat_sayisi'], errors='coerce').fillna(5)

# Fay uzaklığı türetilmiş özellikler
df['fay_yakinligi'] = 25 - df['fay_uzakligi'].clip(0, 25)  # Tersine çevir (yakın = yüksek)
df['fay_risk_skoru'] = df['fay_yakinligi'] * 2  # Ağırlıklandır

# Risk faktörü (İLÇE YOK, FAY UZAKLIĞI VAR)
df['risk_faktoru'] = (
    df['bina_yasi_sayi'] * 0.4 +          # Bina yaşı
    df['yumusak_kat_enc'] * 20 +           # Yumuşak kat riski
    (1 - df['nizami_enc']) * 15 +          # Nizami olmama riski
    (1 - df['simetrik_enc']) * 12 +        # Asimetrik yapı riski
    df['kapali_cikma_enc'] * 8 +           # Kapalı çıkma riski
    df['kat_sayisi_num'] * 2 +             # Kat sayısı riski
    df['fay_yakinligi'] * 1.5              # FAY YAKLIĞI RİSKİ
)

# Etkileşim özellikleri
df['yas_x_kat'] = df['bina_yasi_sayi'] * df['kat_sayisi_num']
df['fay_x_yas'] = df['fay_yakinligi'] * df['bina_yasi_sayi'] / 40
df['fay_x_kat'] = df['fay_yakinligi'] * df['kat_sayisi_num'] / 10
df['yapisal_risk'] = (1 - df['nizami_enc']) + (1 - df['simetrik_enc']) + df['yumusak_kat_enc']

print("\n✅ Özellikler (İLÇE/MAHALLE OLMADAN):")
print("-" * 50)

# =============================================================================
# MODEL ÖZELLİKLERİ (İLÇE YOK!)
# =============================================================================

feature_columns = [
    # Bina özellikleri
    'bina_yasi_sayi',      # Bina yaşı
    'kat_sayisi_num',      # Kat sayısı
    'yumusak_kat_enc',     # Yumuşak kat var mı
    'kapali_cikma_enc',    # Kapalı çıkma var mı
    'nizami_enc',          # Nizami düzende mi
    'simetrik_enc',        # Simetrik yapı mı
    
    # Fay uzaklığı özellikleri (İLÇE YERİNE)
    'fay_uzakligi',        # Fay hattına uzaklık (km)
    'fay_yakinligi',       # Fay yakınlığı (tersine)
    'fay_risk_skoru',      # Fay risk skoru
    
    # Türetilmiş özellikler
    'risk_faktoru',        # Bileşik risk
    'yas_x_kat',           # Yaş x Kat etkileşimi
    'fay_x_yas',           # Fay x Yaş etkileşimi
    'fay_x_kat',           # Fay x Kat etkileşimi
    'yapisal_risk',        # Yapısal risk skoru
]

print(f"📋 Toplam özellik sayısı: {len(feature_columns)}")
print("\n🏠 Bina Özellikleri:")
for f in feature_columns[:6]:
    print(f"   • {f}")
print("\n🌋 Fay Uzaklığı Özellikleri:")
for f in feature_columns[6:9]:
    print(f"   • {f}")
print("\n📊 Türetilmiş Özellikler:")
for f in feature_columns[9:]:
    print(f"   • {f}")

# =============================================================================
# MODEL EĞİTİMİ
# =============================================================================
print("\n" + "=" * 70)
print("🤖 MODEL EĞİTİMİ (İLÇE/MAHALLE YOK)")
print("=" * 70)

X = df[feature_columns].copy()
y = df['hasar_skoru'].copy()

mask = X.notnull().all(axis=1) & y.notnull()
X, y = X[mask], y[mask]

print(f"\n📊 Kullanılabilir veri: {len(X)}")
print(f"📊 Sınıf dağılımı: {dict(Counter(y))}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modeller
models = {
    'Logistic Regression': LogisticRegression(C=0.3, max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_split=6,
                                             min_samples_leaf=3, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.08,
                                                     min_samples_split=6, min_samples_leaf=3, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
}

print("\n🔄 5-Fold Cross-Validation Sonuçları:")
print("-" * 55)

results = {}
for name, model in models.items():
    acc = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')
    results[name] = {'acc': acc.mean(), 'acc_std': acc.std(), 'f1': f1.mean(), 'f1_std': f1.std()}
    print(f"{name}:")
    print(f"  Accuracy: {acc.mean():.4f} (+/- {acc.std():.4f})")
    print(f"  F1-Score: {f1.mean():.4f} (+/- {f1.std():.4f})")

best_model_name = max(results, key=lambda k: results[k]['f1'])
print(f"\n🏆 En İyi Model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")

# Ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(C=0.3, max_iter=1000, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_split=6,
                                       min_samples_leaf=3, class_weight='balanced', random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.08,
                                           min_samples_split=6, min_samples_leaf=3, random_state=42)),
    ],
    voting='hard'
)

ens_acc = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='accuracy')
ens_f1 = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='f1_weighted')
print(f"\nEnsemble: Accuracy={ens_acc.mean():.4f}, F1={ens_f1.mean():.4f}")

if ens_f1.mean() > results[best_model_name]['f1']:
    final_model = ensemble
    best_model_name = 'Ensemble'
else:
    final_model = models[best_model_name]

print(f"✅ Seçilen Model: {best_model_name}")

# =============================================================================
# FİNAL DEĞERLENDİRME
# =============================================================================
print("\n" + "=" * 70)
print("📈 FİNAL MODEL DEĞERLENDİRMESİ")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(f"\n📊 Test Sonuçları:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\n📋 Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Düşük Risk', 'Orta Risk', 'Yüksek Risk'], zero_division=0))

# =============================================================================
# ÖZELLİK ÖNEMLİLİĞİ
# =============================================================================
print("\n" + "=" * 70)
print("📊 ÖZELLİK ÖNEMLİLİĞİ")
print("=" * 70)

rf_imp = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_imp.fit(X_scaled, y)

feat_imp = pd.DataFrame({
    'Özellik': feature_columns,
    'Önemlilik': rf_imp.feature_importances_
}).sort_values('Önemlilik', ascending=False)

print("\n🔑 Özellik Önemlilik Sıralaması:")
for _, row in feat_imp.iterrows():
    bar = '█' * int(row['Önemlilik'] * 50)
    if 'fay' in row['Özellik']:
        emoji = "🌋"
    elif row['Özellik'] in ['risk_faktoru', 'yas_x_kat', 'yapisal_risk']:
        emoji = "📊"
    else:
        emoji = "🏠"
    print(f"  {emoji} {row['Özellik']:20s}: {row['Önemlilik']:.4f} {bar}")

# Fay özellikleri toplam önemliliği
fay_onem = feat_imp[feat_imp['Özellik'].str.contains('fay')]['Önemlilik'].sum()
bina_onem = feat_imp[~feat_imp['Özellik'].str.contains('fay') & 
                      ~feat_imp['Özellik'].isin(['risk_faktoru', 'yas_x_kat', 'yapisal_risk'])]['Önemlilik'].sum()

print(f"\n📈 Kategori Bazında Önemlilik:")
print(f"  🌋 Fay Uzaklığı Özellikleri: {fay_onem:.2%}")
print(f"  🏠 Bina Özellikleri: {bina_onem:.2%}")

# =============================================================================
# OVERFİTTİNG KONTROLÜ
# =============================================================================
print("\n" + "=" * 70)
print("🔍 OVERFİTTİNG KONTROLÜ")
print("=" * 70)

train_score = final_model.score(X_train, y_train)
test_score = final_model.score(X_test, y_test)
fark = abs(train_score - test_score)

print(f"Train: {train_score:.4f} | Test: {test_score:.4f} | Fark: {fark:.4f}")
print("✅ Model dengeli!" if fark < 0.15 else "⚠️ Overfitting riski!")

# =============================================================================
# TAHMİN FONKSİYONU
# =============================================================================
print("\n" + "=" * 70)
print("🔮 YENİ BİNA TAHMİN SİSTEMİ")
print("=" * 70)

def deprem_risk_tahmin(bina_yasi, kat_sayisi, fay_uzakligi_km,
                       yumusak_kat='Hayır', kapali_cikma='Hayır',
                       nizami='Evet', simetrik='Evet'):
    """
    Bina için deprem risk tahmini yapar.
    
    Parametreler:
    - bina_yasi: '0-5', '6-10', '11-20', '21-30', '30+'
    - kat_sayisi: 1-15 arası
    - fay_uzakligi_km: Fay hattına uzaklık (km)
    - yumusak_kat: 'Evet' veya 'Hayır'
    - kapali_cikma: 'Evet' veya 'Hayır'
    - nizami: 'Evet' veya 'Hayır'
    - simetrik: 'Evet' veya 'Hayır'
    """
    # Özellikleri hazırla
    bina_yasi_map = {'0-5': 2.5, '6-10': 8, '11-20': 15, '21-30': 25, '30+': 40}
    yas_num = bina_yasi_map.get(bina_yasi, 15)
    yumusak_num = 1 if yumusak_kat == 'Evet' else 0
    kapali_num = 1 if kapali_cikma == 'Evet' else 0
    nizami_num = 1 if nizami == 'Evet' else 0
    simetrik_num = 1 if simetrik == 'Evet' else 0
    
    fay_yakinligi = 25 - min(fay_uzakligi_km, 25)
    fay_risk = fay_yakinligi * 2
    
    risk = (yas_num * 0.4 + yumusak_num * 20 + (1-nizami_num) * 15 +
            (1-simetrik_num) * 12 + kapali_num * 8 + kat_sayisi * 2 +
            fay_yakinligi * 1.5)
    
    yas_x_kat = yas_num * kat_sayisi
    fay_x_yas = fay_yakinligi * yas_num / 40
    fay_x_kat = fay_yakinligi * kat_sayisi / 10
    yapisal_risk = (1 - nizami_num) + (1 - simetrik_num) + yumusak_num
    
    features = np.array([[
        yas_num, kat_sayisi, yumusak_num, kapali_num, nizami_num, simetrik_num,
        fay_uzakligi_km, fay_yakinligi, fay_risk,
        risk, yas_x_kat, fay_x_yas, fay_x_kat, yapisal_risk
    ]])
    
    features_scaled = scaler.transform(features)
    pred = final_model.predict(features_scaled)[0]
    
    risk_kat = {0: 'DÜŞÜK RİSK ✅', 1: 'ORTA RİSK ⚠️', 2: 'YÜKSEK RİSK 🔴'}
    
    return {
        'risk_kategorisi': risk_kat[pred],
        'fay_uzakligi_km': fay_uzakligi_km,
        'risk_skoru': risk
    }

# İlçe/mahalle'den fay uzaklığı hesaplama yardımcı fonksiyon
def ilce_fay_uzakligi(ilce, mahalle=None):
    """İlçe ve mahalle bilgisinden fay uzaklığını hesaplar"""
    lat, lon = koordinat_bul(ilce, mahalle)
    return fay_uzakligi_hesapla(lat, lon)

# Örnek tahminler
print("\n📍 ÖRNEK TAHMİNLER:")
print("-" * 60)

ornekler = [
    # (bina_yasi, kat, fay_uzakligi, yumusak, kapali, nizami, simetrik, aciklama)
    ('0-5', 3, 0.5, 'Hayır', 'Hayır', 'Evet', 'Evet', 'Yeni bina, faya çok yakın'),
    ('11-20', 5, 6.0, 'Hayır', 'Hayır', 'Hayır', 'Evet', 'Orta yaşlı, faya yakın'),
    ('30+', 6, 9.0, 'Evet', 'Evet', 'Hayır', 'Hayır', 'Eski ve riskli, faya yakın'),
    ('6-10', 4, 20.0, 'Hayır', 'Hayır', 'Evet', 'Evet', 'Güvenli bina, faya uzak'),
]

for yas, kat, fay, yum, kap, niz, sim, aciklama in ornekler:
    sonuc = deprem_risk_tahmin(yas, kat, fay, yum, kap, niz, sim)
    print(f"\n🏠 {aciklama}")
    print(f"   Bina: {yas} yıllık, {kat} katlı")
    print(f"   🌋 Fay uzaklığı: {fay:.1f} km")
    print(f"   📊 Risk skoru: {sonuc['risk_skoru']:.2f}")
    print(f"   🎯 Sonuç: {sonuc['risk_kategorisi']}")

# İlçe bazlı örnek
print("\n" + "-" * 60)
print("📍 İLÇE BAZLI ÖRNEK (Fay uzaklığı otomatik hesaplanır):")
print("-" * 60)

ilce_ornekleri = [
    ('Belen', 'Fatih'),
    ('İskenderun', 'İsmet İnönü'),
    ('Arsuz', 'Karaağaç'),
]

for ilce, mahalle in ilce_ornekleri:
    fay_uzk = ilce_fay_uzakligi(ilce, mahalle)
    sonuc = deprem_risk_tahmin('11-20', 5, fay_uzk, 'Hayır', 'Hayır', 'Evet', 'Evet')
    print(f"\n📍 {ilce} - {mahalle}")
    print(f"   🌋 Hesaplanan fay uzaklığı: {fay_uzk:.2f} km")
    print(f"   🎯 Aynı bina için risk: {sonuc['risk_kategorisi']}")

# =============================================================================
# ÖZET
# =============================================================================
print("\n" + "=" * 70)
print("💾 MODEL ÖZETİ")
print("=" * 70)

print(f"""
📌 Model Bilgileri:
   - Model: {best_model_name}
   - Özellik Sayısı: {len(feature_columns)}
   - ❌ İLÇE/MAHALLE: Modelde YOK (doğrudan kullanılmıyor)
   - ✅ FAY UZAKLIĞI: Modelde VAR (lokasyon bilgisi olarak)

📊 Model Özellikleri:
   🏠 Bina Özellikleri (6 adet):
      - Bina yaşı, Kat sayısı
      - Yumuşak kat, Kapalı çıkma
      - Nizami düzen, Simetrik yapı
   
   🌋 Fay Uzaklığı Özellikleri (3 adet):
      - fay_uzakligi: Doğrudan km mesafe
      - fay_yakinligi: Tersine çevrilmiş (yakın = yüksek)
      - fay_risk_skoru: Ağırlıklandırılmış risk
   
   📊 Türetilmiş Özellikler (5 adet):
      - risk_faktoru, yas_x_kat
      - fay_x_yas, fay_x_kat, yapisal_risk

📈 Performans:
   - Test Accuracy: {test_score:.4f}
   - Overfitting: {'Yok ✅' if fark < 0.15 else 'Var ⚠️'}

💡 Kullanım:
   1. Doğrudan: deprem_risk_tahmin(bina_yasi, kat, fay_km, ...)
   2. İlçe ile: ilce_fay_uzakligi(ilce, mahalle) -> fay_km
""")

print("=" * 70)
print("✅ MODEL BAŞARIYLA OLUŞTURULDU!")
print("=" * 70)

# Model kaydet
import pickle
with open('/home/enesbinmar/Masaüstü/deprem_risk_tahmin_modeli/model_v4_sadece_fay.pkl', 'wb') as f:
    pickle.dump({
        'model': final_model, 
        'scaler': scaler,
        'feature_columns': feature_columns,
        'ilce_koordinatlari': ilce_koordinatlari,
        'mahalle_koordinatlari': mahalle_koordinatlari,
        'fay_hatti': fay_hatti
    }, f)
print("\n💾 Model 'model_v4_sadece_fay.pkl' olarak kaydedildi!")
