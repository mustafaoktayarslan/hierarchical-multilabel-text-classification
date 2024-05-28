
# Hiyerarşik Multi Label Metin Sınıflandırma
Bu depo, hiyerarşik metin sınıflandırması için bir BERT modelini eğitmeye yönelik bir komut dosyalarını içerir. Kod, Root modelin ve alt sınıf modellerinin eğitimi dahil olmak üzere farklı eğitim yapılandırmalarına izin verir. Kod, veri ön işleme, model eğitimi, doğrulama ve doğrulama kaybına dayalı olarak en iyi modeli kaydetme, test etme ve pipeline ile tahminleme işlemlerini gerçekleştirecek şekilde tasarlanmıştır. 




## Kurulum

Gereksinimlerin Kurulumu

Betiği çalıştırmak için gereken tüm kütüphanelerin yüklü olduğundan emin olun. Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:

```bash
pip install -r requirements.txt

```


## Kullanım ve İçerik
Veri seti  Excel dosyası (örneğin, .xlsx formatında) olmalıdır. Yapı 2 kırılımlı  hiyerarşik yapıya uygun kodlanmıştır. Gerek duyulduğu takdirde alt kırılımlar eklenebilir, kodlar bu duruma göre güncellenmelidir. Veri seti aşağıdaki gibi sütunları içermelidir:
- Domain: Ana sınıf etiketleri
- area: Alt sınıf etiketleri
- Abstract: Metin verisi (sınıflandırılacak metinler)
Python dosyaları aynı dizinde bulunmalıdır. (creator.py, train.py, eval.py, predict.py)
- Veri setine buradan erişebilirsiniz. --> [WOS Veri Seti](https://data.mendeley.com/datasets/9rw3vkcfy4/2)

Önerilen yapı aşağıdaki gibidir. Scriptler proje-dizininde çalıştırılmalıdır. Bu dosya düzeniyle çalışıldığında gerekli olan değerleri vermek çalıştırmak için yeterlidir.
```bash
proje-dizini/
│
├── creator.py
├── train.py
├── test.py
├── eval.py
├── predict.py
│
├── data/
│    ├── Data.xlsx
│    ├── df_test.xlsx
│    ├── df_train.xlsx
│    ├── Data.json
│    └── ...
│
└──models/
     ├── model1.pth
     ├── model2.pth
     └── ...

```

- Train çalıştığında oluşan .json dosyası genel hiyerarşiyi gösterir.
- Sınıflar ve hiyerarşi json datadan alınabilir. Kodda creator.py dosyasından çalışma zamanında çekilmiştir.
 ### **creator.py**
    
İçerisindeki DataProcessor sınıfı, veri setini işleyerek eğitim ve test setlerine ayırmayı, veriyi kategorilere ayırarak etiketlemeyi ve JSON formatında metadata oluşturmayı sağlar. Bu sınıf, özellikle  modeller için veriyi hazırlama sürecinde kullanışlıdır. Veriyi temizleme, kategorilere ayırma, etiketleme ve setlere ayırma işlemlerini otomatik hale getirir.

    Veri Okuma ve Ön İşleme:
        Giriş Excel dosyası okunur ve DataFrame olarak yüklenir.
        Metinler küçük harfe dönüştürülür ve gereksiz boşluklar temizlenir.

    Kategorilerin ve Alanların Belirlenmesi:
        Veri setindeki ana kategoriler (Domain) ve alt kategoriler (area) çıkarılır ve sıralanır.
        Metin verileri Abstract sütunundan alınır.
        Ana kategoriler (root classes) ve alt kategoriler (sub-classes) bir JSON formatında yapılandırılır.

    Sınıf Kodlama:
        Ana kategoriler etiketlenir ve kodlanır (label2id). [dataframede encoded sütunu]

    Eğitim ve Test Setlerinin Oluşturulması:
        Veri seti, belirtilen test boyutuna göre eğitim ve test setlerine ayrılır. Dosyalarda oluşturulan df_test eğitimler için df_test kullanılmalıdır. [df_new] [df_test] 
        Stratified split (katmanlı ayırma) uygulanarak veri dengesi korunur.

    Alt Kategorilerin İşlenmesi:
        Ana kategorilere göre veri setleri alt kategorilere ayrılır ve kodlanır.
        Her bir alt kategori için yeni DataFrame'ler oluşturulur.

    Çıktıların Kaydedilmesi:
        Eğitim ve test setleri Excel formatında kaydedilir.
        Sınıf bilgileri JSON formatında kaydedilir.

Örnek Çıktılar

Betik çalıştırıldığında aşağıdaki dosyalar oluşturulacaktır:

    train.xlsx: Eğitim veri seti.
    test.xlsx: Test veri seti.
    Data.json: Sınıf bilgilerini içeren JSON dosyası.


 ### **train.py**
        
Bu kod, bir metin sınıflandırma modelinin eğitilmesi için bir dizi sınıf ve fonksiyon içerir. Özellikle BERT tabanlı modeller kullanarak doğal dil işleme (NLP) görevleri için metin sınıflandırma yapmayı hedefler. Kodun genel amacı, belirli bir veri kümesi üzerinde bir kök model (root model) ve alt sınıf modelleri (subclass models) eğitmek ve bu modelleri belirli dosya yollarına kaydetmektir.
    
Komut satırından bu kodu çalıştırarak modeli eğitebilirsiniz. Aşağıda bazı komut satırı parametreleri verilmiştir:

 


    - model_name: Model ve tokenizer adı.[GEREKLİ]
    - data_path: Veri dosyasının yolu. [GEREKLİ]
    - epoch: Epoch sayısı. [Default=4]
    - batch: Batch boyutu. [Default=16]
    - max_len: Maksimum token uzunluğu. [Default=128]
    - lr: Öğrenme oranı. [Default=3e-5]
    - weight_decay: Ağırlık bozunumu değeri. [Default= 3e-4]
    - warmup: Isınma adımları. [Default=0.2]
    - seed: Rastgelelik tohumu. [Default=42]
    - save_directory: Modelin kaydedileceği dizin. [Default=Mevcut dizin]
    - training_type: Eğitim türü (kök, alt sınıf veya yalnızca alt sınıf). [Default=root]
    - cls_id: Alt sınıf kimliği. [Default=0]
    - device: Kullanılacak cihazı belirtir (cpu veya cuda).[Default=cuda]

***Root model eğitimi***
```bash
  python train.py -model_name 'bert-base-uncased' -data_path ./data/Data.xlsx -epoch 4 -batch 16 -max_len 128 -lr 3e-5 -weight_decay 3e-4 -warmup 0.2 -seed 42 -save_directory /path/to/save -training_type root -device cuda
```
***Alt sınıf model eğitimi***

Diğer eğitim tipleri için ise -training_type subclass veya -training_type only parametrelerini kullanabilirsiniz. -cls_id parametresi sadece only tipi eğitimde gereklidir ve alt sınıf ID'sini belirtir.
```bash
  python train.py -model_name bert-base-uncased -data_path ./data/Data.xlsx -epoch 4 -batch 16 -max_len 128 -lr 3e-5 -weight_decay 3e-4 -warmup 0.2 -seed 42 -save_directory /path/to/save -training_type subclass -device cuda
```
```bash
  python train.py -model_name bert-base-uncased -data_path ./data/Data.xlsx -epoch 4 -batch 16 -max_len 128 -lr 3e-5 -weight_decay 3e-4 -warmup 0.2 -seed 42 -save_directory /path/to/save -training_type only -cls_id 0 -device cuda
```
### eval.py
        
Bu Python betiği, bir modelin performansını değerlendirmek,test etmek için kullanılır. Özellikle, kök veri kümesi veya alt sınıf veri kümesi üzerinde modelin doğruluğunu, hassasiyetini, geri çağırmasını ve F1 skorunu hesaplar. Toplu bir veri seti verilmesi gerekir.
    
**Root Veri Kümesi İçin Değerlendirme**

Root veri kümesi üzerinde model performansını değerlendirmek için aşağıdaki komutu çalıştırın:

```bash
 python evaluate.py -data_path ./data/Data.xlsx -model_path './models/Root_Model.pth' -model_name bert-base-uncased -eval_type root -device cuda

```
    - data_path: Veri kümesinin dosya yolunu belirtir. [GEREKLİ]
    - model_path: Değerlendirilecek modelin dosya yolunu belirtir. [GEREKLİ] 
    - model_name: Kullanılan modelin adını belirtir. [GEREKLİ]
    - eval_type: Değerlendirme türünü belirtir. Bu değer "root" olmalıdır. [Default=root]
    - device: Kullanılacak cihazı belirtir (cpu veya cuda).[Default=cuda]
    
    
**Alt Sınıf Veri Kümesi İçin Değerlendirme**

Alt sınıf veri kümesi üzerinde model performansını değerlendirmek için aşağıdaki komutu çalıştırın:

- Not: sınıf sıralamsı --> ['CS', 'Civil', 'ECE', 'MAE', 'Medical', 'Psychology', 'biochemistry']

```bash
 python evaluate.py -data_path ./data/Data.xlsx -model_path models/SModel_CS.pth -model_name model_adı -eval_type sub -cls_id 0 -device cuda

```
    - data_path: Veri kümesinin dosya yolunu belirtir. [GEREKLİ]
    - model_path: Değerlendirilecek modelin dosya yolunu belirtir. [GEREKLİ] 
    - model_name: Kullanılan modelin adını belirtir. [GEREKLİ]
    - eval_type: Değerlendirme türünü belirtir. Bu değer "sub" olmalıdır. [Default=root]
    - cls_id: Alt sınıfın kimliğini belirtir. Bu, alt sınıf lar listesindeki sınıfın indeksi olmalıdır. [Default=0] 
    - device: Kullanılacak cihazı belirtir (cpu veya cuda).[Default=cuda]
    
### predict.py
        
Metin sınıflandırma pipeline'ınızı çalıştırmak ve kullanıcıdan alınan metin girdilerini sınıflandırmak için kullanılır. Bu dosya, kök sınıflandırma modeli ve alt sınıflandırma modelleri kullanarak metinlerin hangi sınıfa ait olduğunu tahmin eder. Bu süreç, kullanıcının komut satırından girdiği metni alarak belirli sınıflara otomatik olarak etiketler.
    
**Metinleri Tahminleme**

    - test_data: Test verisi dosyasının yolu (Excel formatında test datası olmalıdır). (Opsiyonel)
    - data_path: Veri çerçevesi dosya yolu. [GEREKLİ]
    - root_model_path: Kök model yolu. [GEREKLİ]
    - sub_models_paths: Alt modellerin yolları. Sıralı olarak girin. Default=['./models/SModel_CS', './models/SModel_Civil', './models/SModel_ECE', './models/SModel_MAE','./models/SModel_Medical', './models/SModel_Psychology', './models/SModel_biochemistry']
    - model_name: Model ve tokenizer adı. [Default=bert-base-uncased]
    - device: Kullanılacak cihaz (cpu veya cuda). [Default=cuda]


Aşağıdaki komutu çalıştırarak metin tahminlerini başlatabilirsiniz:
- sub_models_paths opsiyonel olarak verilebilir. Tüm alt sınıflar sırayla verilmelidir.
```bash
 python <script_adı> -data_path <veri_yolu> -root_model_path <kök_model_yolu> -sub_models_paths <alt_model_yolları> -model_name <model_adı> -device <cihaz>
```
Kullanıcıdan metin girişi alarak çalıştırmak için örnek:

```bash
  python predict.py -data_path ./data/Data.xlsx -root_model_path ./models/Root_Model.pth -sub_models_paths ./models/SModel_CS.pth ./models/SModel_Civil.pth ./models/SModel_ECE.pth ./models/SModel_MAE.pth ./models/SModel_Medical.pth ./models/SModel_Psychology.pth ./models/SModel_biochemistry.pth -model_name bert-base-uncased -device cuda
``` 
Test verisi ile çalıştırmak için: örnek:
 - Not: Verilen test datası üzerinde tahminleme yapar ve test sonuçlarını oluşturur. Excel dosyası ile test verisi verilmesi gerekir
```bash
  python predict.py -test_data ./data/df_test.xlsx -data_path ./data/Data.xlsx -root_model_path ./models/Root_Model.pth -sub_models_paths ./models/SModel_CS.pth ./models/SModel_Civil.pth ./models/SModel_ECE.pth ./models/SModel_MAE.pth ./models/SModel_Medical.pth ./models/SModel_Psychology.pth ./models/SModel_biochemistry.pth -model_name bert-base-uncased -device cuda

``` 
