
# NST Takımı Kaynak Kodları
#### TEKNOFEST 2023 Ulaşımda Yapay Zeka Yarışması Finalisti  NST Takımı Kaynak Kodları
Bu çalışmada, uçan araçlarla alınan görüntülerde taşıtlar, insanlar ve iniş-kalkış alanlarının tespiti için yapay zeka tabanlı bir nesne tespit sistemi kullanılmıştır. YOLOv6 mimarisiyle oluşturulan model, farklı ortamlarda çalışabilen ve iki paralel modelle farklı boyutlardaki nesneleri tanıyabilen bir "KASK (Karar Seçme Katmanı)" kullanmaktadır. Veriler, yüksek çözünürlüklü olduğundan "sliding-window" algoritması kullanılarak uygun boyutlara bölünmektedir. Bu sistem sayesinde uçan araçların iniş-kalkış yapabileceği yerler ve zemindeki araç-insan durumu canlı ortamda analiz edilebilmektedir.

## Takım Üyeleri
<div style="display: grid; grid-template-columns: auto auto auto auto;">
    <div>
        <a href="https://github.com/AAhmetDurmaz" style="width: 100px;">
            <div style="width: 100%; display: flex; justify-content: center;">
                <img src="https://github.com/AAhmetDurmaz.png" width="75" height="75" style="border-radius: 150px;"/>
            </div>
                <center>A. Ahmet Durmaz</center>
        </a>
        <center>Üye</center>
    </div>
    <div>
        <a href="https://github.com/barisazar" style="width: 100px;">
            <div style="width: 100%; display: flex; justify-content: center;">
                <img src="https://github.com/barisazar.png" width="75" height="75" style="border-radius: 150px;"/>
            </div>
                <center>Barış Azar</center>
        </a>
        <center>Üye</center>
    </div>
    <div>
        <a href="https://github.com/YusufKizilgedik" style="width: 100px;">
            <div style="width: 100%; display: flex; justify-content: center;">
                <img src="https://github.com/YusufKizilgedik.png" width="75" height="75" style="border-radius: 150px;"/>
            </div>
                <center>Yusuf Kızılgedik</center>
        </a>
        <center>Üye</center>
    </div>
    <div>
        <a href="https://github.com/mregungor" style="width: 100px;">
            <div style="width: 100%; display: flex; justify-content: center;">
                <img src="https://github.com/mregungor.png" width="75" height="75" style="border-radius: 150px;"/>
            </div>
                <center>Dr. Emre Güngör</center>
        </a>
        <center>Danışman</center>
    </div>
</div>

## KASK nedir?
KASK (Karar Seçme Katmanı), iki farklı model senkron şekilde çalışması, girdi ve çıktıların istenilen sonuçlara dönüştürülmesi gerektiğinden ortaya çıkmış bir yapıdır. Sistemde, hiyerarşik olarak en üst katmanında yer alan ve manuel olarak geliştirilen KASK’ın (Karar Seçme Katmanı) görevleri aşağıda belirtilmiştir;
* KASK, dinamik, hızlı bir algoritmadır ve iki modelden gelen verileri senkronize eder.
* KASK, sistemin girdi ve çıktılarından sorumludur ve verilerin sliding-window algoritması yardımı ile parçalanmasını sağlar.
* KASK, görüntü bozunumu olup olmadığını kontrol eder ve bulanıklığı azaltmaya çalışarak modellerin daha doğru sonuçlar vermesini sağlar.
* KASK, her sınıfa özel threshold(eşik değer) değerine göre davranır ve net bir çıktı elde etmeye yardımcı olur.
* KASK, hata engelleyici algoritmalar eklenerek bir nesnenin birden fazla sınıf da işaretlenmesini önler ve iki modelin senkron şekilde çalışmasını sağlar.

## Veri setleri
Başta TEKNOFEST'in özel olarak sağladığı geçmiş yıl verileri olmak üzere,
* [Visdrone 2019](https://github.com/VisDrone/VisDrone-Dataset)
* [VAID](https://github.com/KaiChun-RVL/VAID_dataset)
* Stok videolardan işaretlenerek oluşturulmuş veriler
karıştırılarak hibrit bir veri seti olan NST-v3.3 oluşturulmuştur.

## YOLOv6-L Eğitimi
[YOLOv6-Large modeli](https://github.com/meituan/YOLOv6/releases/tag/0.4.0) meituan tarafından geliştirilmiştir. Açık olarak erişilebilmektedir.

Bu model kullanılarak NST-v3.3 veri seti ile 50 epoch eğitilmiştir. Eğitilmiş modele [buradan](https://drive.google.com/drive/folders/13V1o9SvbDddMvR-G1yIcLyu4Yhx4rJNs?usp=sharing) ulaşabilirsiniz.
