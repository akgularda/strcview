# sTRCview

sTRCview, literatür taraması ve bibliyometri için bir masaüstü uygulamasıdır. Kullanıcılar arama sonuçlarını içeri alır, yayınların başlık ve özetlerini değerlendirir ve tarama kararlarını dışa aktarır. [Sürümleri indir](https://github.com/akgularda/strcview/releases).

## Windows’ta açılış

`sTRCview-windows-x64.zip` dosyasını yazılabilir bir klasöre çıkar ve içindeki `sTRCview.exe` dosyasını aç. Paket Python çalışma ortamını içerir ve Windows x64 ile Microsoft Edge WebView2 Runtime kullanır. İlk açılışta Windows’un WebView2 çalışma ortamına ihtiyacı varsa [Microsoft WebView2](https://developer.microsoft.com/microsoft-edge/webview2/) sayfasından kurulabilir.

Yayınlar, protokol, ilk örneklerin sırası ve kararlar uygulama klasöründeki `data/workspace.json` dosyasına otomatik kaydedilir. Taşırken tüm klasörü taşı veya **Ayarlar → Çalışma alanı** bölümünden JSON yedeğini indir. Laya, ilk kullanımda gerekli model ağırlıklarını `data/models` altına indirir. Ağırlıklar hazır olduğunda değerlendirme yerelde çalışır.

## Mac’te açılış

Apple Silicon (M serisi) için `sTRCview-macos-arm64.zip` dosyasını aç ve `sTRCview.app` uygulamasını **Uygulamalar** klasörüne taşı. İlk açılışta macOS, Apple tarafından noter onayı verilmemiş geliştirici derlemesini engellerse, açmayı denedikten sonra **Sistem Ayarları → Gizlilik ve Güvenlik → Yine de Aç** yolunu kullan. Veriler ve model önbelleği `~/Library/Application Support/sTRCview` altında saklanır.

## Çalışma akışı

1. **Kütüphane → Dosya al** bölümünden yayınları yükle. Özel tabloda çalışma sayfasını ve sütunları doğrula. **Kopyalar** bölümünden eşleşen kayıtları yan yana incele.
2. **Ayarlar → Araştırma protokolü** bölümünde soruyu ve ölçütleri kaydet.
3. **Tarama** bölümünde başlangıç örneklerini seç. Laya en fazla 32 adayı puanlayarak yüksek, düşük ve karar eşiğine yakın sekiz örnek seçer. Sekiz veya daha az kayıt varsa tümünü gösterir. Laya yüklenemezse kaynak çeşitliliğine göre seçilen örneklerle devam edebilirsin.
4. Her makaleyi **İlgili**, **İlgisiz** veya **Kararsız** olarak işaretle. İlk örneklerden sonra en az bir ilgili ve bir ilgisiz karar varsa Laya, kendi karar katmanını bu etiketlerle günceller ve kalan kayıtları sıralar. İlk sıralama, koleksiyonun büyüklüğüne göre zaman alabilir. **Kütüphane → Yayınlar** bölümünde inceleme önceliğine göre sırala, kararları düzelt veya kararsız kayıtları yeniden aç.
5. **Raporlar** bölümünden tarama akışını, bibliyometriyi ve hakem uyumunu incele. Excel, CSV, Scopus biçimli CSV ve tarama raporu aynı bölümden indirilebilir.

Web of Science’ta Tagged Text, TSV veya CSV, Scopus’ta CSV veya RIS dışa aktar. Elindeki özel Excel, CSV veya TSV tablosunda her satıra bir yayın koy. Başlık zorunludur. Özeti, DOI’yi, yazarları, yılı ve anahtar kelimeleri de eklemek tarama ve kopya incelemesini kolaylaştırır. Uygulama başlık ve özet taraması yapar. Tam metin değerlendirmesini, uygun görünen yayınların PDF’leri üzerinde ayrıca yürütürsün. İkinci hakem kararları DOI veya yayın kimliği ve karar sütunu içeren CSV ile karşılaştırılabilir.

İsteğe bağlı toplu model önerileri **Ayarlar → Model** bölümündedir. Ollama için yerel hizmet ve `llama3.2` modeli gerekir. Unpaywall sorgusunu etkinleştirmek için başlatmadan önce `STRCVIEW_UNPAYWALL_EMAIL` ortam değişkenini ayarla. Sorgu düğmesi seçili DOI ve e-posta adresini Unpaywall’a gönderir. [Unpaywall API](https://unpaywall.org/api)

## Kaynak koddan çalıştırma

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[dev,desktop]"
.\.venv\Scripts\python.exe src\desktop.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

Windows’ta tarayıcıyla geliştirme için `run.bat` kullanılabilir. Her platformun paketi kendi işletim sisteminde `python -m PyInstaller desktop.spec` ile üretilir. Apple Silicon paketi için macOS ve ARM64 Python kullan.
