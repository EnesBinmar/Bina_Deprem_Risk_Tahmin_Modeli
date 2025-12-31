# 🏗️ Bina Deprem Risk Tahmin Modeli

Hatay bölgesindeki binaların deprem risk skorunu tahmin eden makine öğrenmesi tabanlı sistem. Fay hattına uzaklık, bina yaşı, kat sayısı gibi özellikleri kullanarak risk değerlendirmesi yapar.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Detayları](#model-detayları)
- [Veri Yapısı](#veri-yapısı)
- [Teknolojiler](#teknolojiler)

## ✨ Özellikler

- 🎯 **Hassas Risk Tahmini**: Çoklu makine öğrenmesi modellerinin birleşimiyle yüksek doğruluk
- 📍 **Fay Hattı Analizi**: Doğu Anadolu Fay Hattı'na olan uzaklığa göre risk hesaplama
- 🌐 **Web Arayüzü**: Flask tabanlı kullanıcı dostu web arayüzü
- 📊 **Çoklu Parametre**: Bina yaşı, kat sayısı, zemin türü, yapı kalitesi gibi çeşitli faktörler
- 🏘️ **Lokasyon Bazlı**: Hatay'ın 14 ilçesi ve mahalle bazında detaylı analiz

## 🚀 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- pip paket yöneticisi

### Adımlar

1. Repoyu klonlayın:
```bash
git clone https://github.com/EnesBinmar/Bina_Deprem_Risk_Tahmin_Modeli.git
cd Bina_Deprem_Risk_Tahmin_Modeli
```

2. Gerekli paketleri yükleyin:
```bash
pip install flask numpy pandas scikit-learn
```

3. Modeli eğitin (isteğe bağlı):
```bash
python deprem_risk_modeli_v4_sadece_fay.py
```

4. Web uygulamasını başlatın:
```bash
python web_app.py
```

veya Linux/Mac için:
```bash
bash baslat.sh
```

5. Tarayıcınızda açın:
```
http://localhost:5000
```

## 💻 Kullanım

### Web Arayüzü

1. Web uygulamasını başlatın
2. Bina bilgilerini girin:
   - İlçe ve mahalle seçimi
   - Bina yaşı
   - Kat sayısı
   - Zemin türü
   - İnşaat yılı
   - Yapı kalitesi
3. "Risk Hesapla" butonuna tıklayın
4. Risk skoru ve değerlendirmeyi görüntüleyin

### Python Modülü Olarak

```python
from deprem_risk_modeli_v4_sadece_fay import predict_risk

# Örnek veri
bina_bilgileri = {
    'ilce': 'İskenderun',
    'mahalle': 'İsmet İnönü',
    'bina_yasi': 25,
    'kat_sayisi': 5,
    'zemin_turu': 'Orta sert',
    'insaat_yili': 1998,
    'yapi_kalitesi': 'Orta'
}

# Risk tahmini
risk_skoru = predict_risk(bina_bilgileri)
print(f"Risk Skoru: {risk_skoru}")
```

## 🔬 Model Detayları

### Kullanılan Modeller

- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVC)**
- **Voting Classifier** (Ensemble)

### Özellikler (Features)

1. **Fay Hattına Uzaklık** (km) - En kritik faktör
2. **Bina Yaşı**
3. **Kat Sayısı**
4. **Zemin Türü** (Yumuşak, Orta sert, Sert)
5. **İnşaat Yılı**
6. **Yapı Kalitesi** (Düşük, Orta, Yüksek)

### Performans Metrikleri

- **Doğruluk (Accuracy)**: ~85-90%
- **F1 Score**: ~0.87
- **Cross-Validation**: 5-fold stratified

## 📊 Veri Yapısı

### Veri Dosyaları

- `data/veri_fay_uzaklikli.csv`: Eğitim verisi (fay uzaklıkları ile)
- `data/Deprem Risk Skoru Veri - Form Yanıtları 1.csv`: Ham form verileri

### Kapsanan Bölgeler

**İlçeler (14):**
İskenderun, Antakya, Arsuz, Defne, Samandağ, Dörtyol, Belen, Kırıkhan, Payas, Erzin, Hassa, Altınözü, Reyhanlı, Yayladağı

**Mahalleler (20+):**
İsmet İnönü, Sakarya, Numune, Mustafa Kemal, Dumlupınar, Denizciler, Yunus Emre, Kurtuluş, Pirireis, Cumhuriyet, ve diğerleri

## 🛠️ Teknolojiler

- **Backend**: Python 3.x, Flask
- **ML Libraries**: scikit-learn, NumPy, pandas
- **Frontend**: HTML, CSS, Bootstrap
- **Veri İşleme**: pandas, NumPy
- **Model**: Random Forest, Gradient Boosting, SVM

## 📁 Proje Yapısı

```
.
├── deprem_risk_modeli_v4_sadece_fay.py  # Ana model dosyası
├── web_app.py                            # Flask web uygulaması
├── baslat.sh                             # Başlatma scripti
├── data/                                 # Veri dosyaları
│   ├── veri_fay_uzaklikli.csv
│   └── Deprem Risk Skoru Veri - Form Yanıtları 1.csv
└── templates/                            # HTML şablonları
    ├── index.html                        # Ana sayfa
    ├── sonuc.html                        # Sonuç sayfası
    └── error.html                        # Hata sayfası
```

## ⚠️ Önemli Notlar

- Bu model akademik ve araştırma amaçlıdır
- Profesyonel deprem risk değerlendirmesinin yerini tutmaz
- Resmi yapı denetimi ve mühendislik analizi gereklidir
- Tahminler yaklaşık değerlerdir ve kesin sonuç değildir

## 📝 Lisans

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.

## 👤 Geliştirici

**Enes Binmar**
- GitHub: [@EnesBinmar](https://github.com/EnesBinmar)

## 🤝 Katkıda Bulunma

1. Bu repoyu fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

## 📧 İletişim

Sorularınız veya önerileriniz için GitHub Issues kullanabilirsiniz.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
