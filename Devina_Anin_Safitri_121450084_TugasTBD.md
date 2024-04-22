---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.13
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="G6bpBZ-qKa5q"}
**Tugas Teknologi Basis Data**

Devina Anin Safitri

121450084

RA
:::

::: {.cell .markdown id="N9rDKqSuKa5r"}
## Kumpulan Data Untuk di Proses
:::

::: {.cell .code id="kKZzadIrKa5r" outputId="8b7154e0-ec51-4692-af6a-f5f3bded8dc8"}
``` python
import numpy as np
import pickle
from pathlib import Path

# Path to the unzipped CIFAR data
data_dir = Path("C:/Users/ASUS/Downloads/cifar-10-python/cifar-10-batches-py")

# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])

print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```

::: {.output .stream .stdout}
    Loaded CIFAR-10 training set:
     - np.shape(images)     (50000, 32, 32, 3)
     - np.shape(labels)     (50000,)
:::
:::

::: {.cell .markdown id="ZVsHIyTnKa5s"}
Pertama, Kode ini mengimpor modul numpy untuk operasi matematika dan
pickle untuk memuat data yang telah di-pickle, serta menggunakan modul
pathlib untuk bekerja dengan path file. Direktori tempat dataset
CIFAR-10 di-unzip ditentukan, dan fungsi unpickle digunakan untuk memuat
data yang telah di-pickle. Kode ini melakukan iterasi melalui semua file
batch dalam direktori data, memuat data menggunakan fungsi unpickle, dan
kemudian melakukan iterasi melalui setiap gambar dalam batch. Setiap
gambar dalam batch di-flatten dan kemudian dibagi menjadi tiga kanal
warna (R, G, B). Setiap kanal di-reshape menjadi array 32x32 dan
kemudian digabungkan menjadi gambar asli menggunakan np.dstack. Gambar
yang telah diproses disimpan dalam list images, dan label yang sesuai
disimpan dalam list labels. Akhirnya, kode mencetak ukuran dari array
images dan labels, menunjukkan bahwa dataset telah berhasil dimuat
dengan 50.000 gambar dan 50.000 label.
:::

::: {.cell .code id="9ruTXl2fKa5s"}
``` python
from pathlib import Path

disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
```
:::

::: {.cell .markdown id="9aXl_kQ0Ka5t"}
Kode tersebut menggunakan modul pathlib untuk menangani path file dalam
Python. Modul pathlib memungkinkan pengembang untuk bekerja dengan path
file dengan cara yang lebih mudah dan intuitif dibandingkan dengan
pendekatan tradisional menggunakan modul os atau os.path. Dalam kode
tersebut, tiga variabel disk_dir, lmdb_dir, dan hdf5_dir didefinisikan
menggunakan Path dari modul pathlib, yang masing-masing
merepresentasikan direktori untuk data disk, lmdb, dan hdf5. Ini
menunjukkan bagaimana pathlib dapat digunakan untuk menyederhanakan
manipulasi path file, memudahkan pengembang dalam membuat kode yang
lebih bersih dan mudah dipahami. Modul ini juga memungkinkan pengembang
untuk melakukan operasi seperti pembuatan direktori, penghapusan, dan
pengecekan keberadaan file atau direktori dengan cara yang lebih
ekspresif dan Pythonic. Dengan kata lain, pathlib menyediakan API yang
konsisten dan mudah digunakan untuk berbagai operasi file dan direktori,
membuatnya menjadi pilihan yang baik untuk pengembangan Python modern
:::

::: {.cell .code id="0J87HxbeKa5t"}
``` python
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)
```
:::

::: {.cell .markdown id="kFHRXEL6Ka5t"}
Kode tersebut menggunakan metode mkdir dari objek Path yang
didefinisikan oleh modul pathlib untuk membuat direktori. Metode ini
memungkinkan pembuatan direktori dengan opsi untuk menangani situasi
ketika direktori induk tidak ada atau direktori sudah ada. Argumen
parents=True memastikan bahwa semua direktori induk yang hilang akan
dibuat seiring kebutuhan, mimik perilaku dari perintah mkdir -p dalam
POSIX. Argumen exist_ok=True mengabaikan FileExistsError jika direktori
sudah ada, memungkinkan kode untuk berjalan tanpa menghasilkan kesalahan
jika direktori yang diinginkan sudah ada. Ini sangat berguna untuk skrip
yang mungkin dijalankan beberapa kali atau dalam lingkungan di mana
kondisi direktori mungkin berubah antara eksekusi. Dengan menggunakan
parents=True dan exist_ok=True, kode menjadi lebih robust dan mudah
digunakan dalam berbagai kondisi
:::

::: {.cell .code id="RZNXJHI4Ka5t"}
``` python
from PIL import Image
import csv

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")

    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
```
:::

::: {.cell .markdown id="lwsvVnnsKa5t"}
Fungsi store_single_disk yang didefinisikan dalam kode ini bertujuan
untuk menyimpan satu gambar sebagai file .png dan labelnya sebagai file
.csv di disk. Fungsi ini menggunakan modul PIL (Python Imaging Library)
untuk bekerja dengan gambar dan modul csv untuk menulis data ke file
CSV.

-   Parameter Fungsi: Fungsi ini menerima tiga parameter: image,
    image_id, dan label. image adalah array gambar dengan dimensi (32,
    32, 3), yang mewakili gambar dengan ukuran 32x32 piksel dan tiga
    kanal warna (R, G, B). image_id adalah ID unik integer untuk gambar,
    dan label adalah label gambar.
-   Penyimpanan Gambar: Menggunakan metode Image.fromarray dari modul
    PIL, gambar dikonversi dari array numpy ke objek gambar PIL.
    Kemudian, metode save digunakan untuk menyimpan gambar sebagai file
    .png di direktori yang ditentukan oleh disk_dir. Nama file gambar
    dibentuk dengan menggabungkan disk_dir dengan image_id dan ekstensi
    .png.
-   Penyimpanan Label: Setelah gambar disimpan, fungsi ini membuka file
    CSV baru dengan nama yang sama dengan image_id dan ekstensi .csv di
    direktori yang sama. File CSV ini digunakan untuk menyimpan label
    gambar. Fungsi menggunakan modul csv untuk menulis label ke dalam
    file CSV. Delimiter yang digunakan adalah spasi, dan karakter kutip
    adalah \|. Opsi quoting=csv.QUOTE_MINIMAL memastikan bahwa label
    hanya dikutip jika mengandung karakter spesial yang memerlukan
    kutipan. Fungsi ini menunjukkan bagaimana menggabungkan operasi
    pengolahan gambar dan penulisan data ke file dengan Python.
:::

::: {.cell .code id="lX9c1rLSKa5t"}
``` python
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
```
:::

::: {.cell .markdown id="RIqAEgdNKa5t"}
Kelas CIFAR_Image yang didefinisikan dalam kode ini dirancang untuk
menangani gambar dari dataset CIFAR-10 dalam format yang lebih fleksibel
dan mudah digunakan. Kelas ini menyimpan gambar sebagai byte dan label,
memungkinkan penggunaan yang lebih efisien dari memori dan memudahkan
operasi pada gambar.

-   Inisialisasi: Konstruktor **init** menerima dua parameter: image dan
    label. image adalah array numpy yang mewakili gambar, dan label
    adalah label yang sesuai dengan gambar tersebut. Dalam konstruktor,
    dimensi gambar disimpan dalam self.size, dan jumlah kanal warna
    disimpan dalam self.channels. Gambar itu sendiri dikonversi menjadi
    byte dan disimpan dalam self.image. Label disimpan langsung dalam
    self.label.
-   Metode get_image: Metode ini mengembalikan gambar dalam format array
    numpy. Pertama, byte gambar yang disimpan dalam self.image
    dikonversi kembali menjadi array numpy menggunakan np.frombuffer.
    Tipe data yang digunakan adalah np.uint8, yang sesuai dengan format
    piksel gambar dalam dataset CIFAR-10. Kemudian, array ini direshape
    menjadi bentuk asli gambar menggunakan self.size dan self.channels.
    Ini memungkinkan penggunaan gambar dalam format yang sama dengan
    yang aslinya, memudahkan operasi pengolahan gambar. Kelas ini
    menunjukkan bagaimana mengoptimalkan penggunaan memori dan
    memudahkan manipulasi gambar dalam Python. Dengan menyimpan gambar
    sebagai byte dan menyediakan metode untuk mengembalikannya ke format
    array numpy.
:::

::: {.cell .code id="ZXU0i7ydKa5u"}
``` python
import lmdb
import pickle

def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10

    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
```
:::

::: {.cell .markdown id="yXRTG-9OKa5u"}
Kode di atas adalah sebuah fungsi Python yang digunakan untuk menyimpan
gambar ke dalam basis data LMDB (Lightning Memory-Mapped Database).
Fungsi store_single_lmdb yang didefinisikan dalam kode ini bertujuan
untuk menyimpan satu gambar dan labelnya ke dalam database LMDB
(Lightning Memory-Mapped Database). LMDB adalah database key-value yang
dirancang untuk penggunaan dalam memori dan disk, menawarkan akses yang
cepat dan efisien terhadap data.

-   Parameter Fungsi: Fungsi ini menerima tiga parameter: image,
    image_id, dan label. image adalah array gambar dengan dimensi (32,
    32, 3), yang mewakili gambar dengan ukuran 32x32 piksel dan tiga
    kanal warna (R, G, B). image_id adalah ID unik integer untuk gambar,
    dan label adalah label gambar.
-   Pembuatan Environment LMDB: Fungsi ini menentukan ukuran map LMDB
    dengan mengalikan ukuran byte dari gambar dengan 10. Kemudian,
    environment LMDB dibuat di direktori yang ditentukan oleh lmdb_dir
    dengan nama single_lmdb dan ukuran map yang ditentukan.
-   Transaksi Penulisan: Fungsi ini membuka transaksi penulisan baru ke
    dalam environment LMDB. Dalam transaksi ini, gambar dan label diubah
    menjadi objek CIFAR_Image (yang diasumsikan telah didefinisikan
    sebelumnya) dan kemudian di-pickle. Kunci untuk menyimpan gambar dan
    label dibuat dengan menggunakan image_id yang diubah menjadi string
    dengan format 8 digit (misalnya, \"00000001\" untuk image_id 1).
    Kunci dan nilai (dalam bentuk byte) kemudian disimpan ke dalam
    database.
-   Penutupan Environment: Setelah gambar dan label disimpan,
    environment LMDB ditutup. Fungsi ini menunjukkan bagaimana
    menggunakan LMDB untuk menyimpan dan mengakses data gambar dan label
    dengan cepat dan efisien.
:::

::: {.cell .code id="nH3sIFNRKa5u"}
``` python
import h5py

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
```
:::

::: {.cell .markdown id="YbN-TkvDKa5u"}
Fungsi store_single_hdf5 yang didefinisikan dalam kode ini bertujuan
untuk menyimpan satu gambar dan labelnya ke dalam file HDF5. HDF5 adalah
format file yang dirancang untuk menyimpan data yang kompleks dan besar,
seperti gambar dan label, dalam format yang efisien dan mudah diakses.

-   Parameter Fungsi: Fungsi ini menerima tiga parameter: image,
    image_id, dan label. image adalah array gambar dengan dimensi (32,
    32, 3), yang mewakili gambar dengan ukuran 32x32 piksel dan tiga
    kanal warna (R, G, B). image_id adalah ID unik integer untuk gambar,
    dan label adalah label gambar.
-   Pembuatan File HDF5: Fungsi ini membuat file HDF5 baru di direktori
    yang ditentukan oleh hdf5_dir dengan nama yang dibentuk dengan
    menggabungkan image_id dan ekstensi .h5.
-   Pembuatan Dataset: Dalam file HDF5 yang baru dibuat, dua dataset
    dibuat: image dan meta. Dataset image dibuat dengan ukuran yang
    sesuai dengan image dan tipe data h5py.h5t.STD_U8BE, yang merupakan
    8-bit unsigned integer. Dataset ini berisi data gambar. Dataset meta
    dibuat dengan ukuran yang sesuai dengan label dan juga tipe data
    h5py.h5t.STD_U8BE, berisi data label. Penutupan File: Setelah
    dataset dibuat, file HDF5 ditutup. Fungsi ini menunjukkan bagaimana
    menggunakan HDF5 untuk menyimpan dan mengakses data gambar dan label
    dengan cepat dan efisien.
:::

::: {.cell .code id="DlgahiYxKa5u"}
``` python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```
:::

::: {.cell .markdown id="_9rjONASKa5u"}
Dalam kode yang diberikan, \_store_single_funcs adalah sebuah dictionary
yang memetakan nama metode penyimpanan ke fungsi yang sesuai. Ini
memungkinkan pemanggilan fungsi penyimpanan berdasarkan nama metode yang
dipilih, sehingga memudahkan eksekusi kode yang dinamis dan fleksibel.

-   Disk: store_single_disk adalah fungsi yang menyimpan gambar sebagai
    file .png dan label sebagai file .csv di disk. Ini adalah metode
    yang sederhana dan mudah digunakan untuk menyimpan data dalam format
    yang langsung dapat diakses dan dibaca oleh manusia.
-   LMDB: store_single_lmdb adalah fungsi yang menyimpan gambar dan
    label ke dalam database LMDB. LMDB adalah database key-value yang
    dirancang untuk penggunaan dalam memori dan disk, menawarkan akses
    yang cepat dan efisien terhadap data. Kecepatan dan efisiensi memori
    adalah kunci utama dari LMDB, tetapi juga memiliki beberapa
    keterbatasan, seperti kebutuhan untuk menentukan ukuran map sebelum
    menulis ke database dan kemungkinan terjadinya lmdb.MapFullError
    jika ukuran map tidak cukup.
-   HDF5: store_single_hdf5 adalah fungsi yang menyimpan gambar dan
    label ke dalam file HDF5. HDF5 adalah format file yang dirancang
    untuk menyimpan data yang kompleks dan besar dalam format yang
    efisien dan mudah diakses. HDF5 memungkinkan akses cepat ke item
    yang diminta, dengan kemampuan untuk mengakses dataset seperti array
    Python, memudahkan operasi seperti indeks, rentang, dan
    pemotongan 1. Pemilihan metode penyimpanan tergantung pada kebutuhan
    spesifik aplikasi, seperti kecepatan akses data, ukuran data, dan
    preferensi pengguna terhadap format file.
:::

::: {.cell .code id="PPt-FXqlKa5u" outputId="016517fb-a2d4-4e97-dc7b-149481643378"}
``` python
from timeit import timeit

store_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.14079929999979868
    Method: lmdb, Time usage: 0.005631300000004558
    Method: hdf5, Time usage: 0.06091969999988578
:::
:::

::: {.cell .markdown id="TudCsivJKa5v"}
Kode tersebut menunjukkan penggunaan modul timeit untuk mengukur waktu
eksekusi dari tiga metode penyimpanan gambar: disk, LMDB, dan HDF5. Ini
dilakukan dengan menjalankan setiap fungsi penyimpanan sekali dan
mengukur waktu yang dibutuhkan untuk menyelesaikannya. Hasilnya disimpan
dalam dictionary store_single_timings untuk analisis lebih lanjut.

-   Metode Disk: Metode ini menyimpan gambar sebagai file .png dan label
    sebagai file .csv di disk. Waktu eksekusi yang dihasilkan adalah
    0.14079929999979868 detik. Metode ini cocok untuk kasus penggunaan
    di mana kemudahan akses dan bacaan manual oleh manusia sangat
    penting.
-   Metode LMDB: Metode ini menyimpan gambar dan label ke dalam database
    LMDB. LMDB menggunakan file yang dipetakan ke memori, memberikan
    performa I/O yang lebih baik dibandingkan dengan metode disk. Waktu
    eksekusi yang dihasilkan adalah 0.005631300000004558 detik. LMDB
    sangat cocok untuk dataset yang sangat besar dan memerlukan akses
    cepat ke data.
-   Metode HDF5: Metode ini menyimpan gambar dan label ke dalam file
    HDF5. HDF5 dirancang untuk menyimpan data yang kompleks dan besar
    dalam format yang efisien dan mudah diakses. Waktu eksekusi yang
    dihasilkan adalah 0.06091969999988578 detik. HDF5 memungkinkan akses
    cepat ke item yang diminta, dengan kemampuan untuk mengakses dataset
    seperti array Python 3. Hasil ini menunjukkan bahwa LMDB menawarkan
    performa I/O yang paling baik dibandingkan dengan metode disk dan
    HDF5, meskipun HDF5 memiliki waktu eksekusi yang sedikit lebih cepat
    dibandingkan dengan disk.
:::

::: {.cell .code id="a3yc1vtfKa5v"}
``` python
def store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")

    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])

def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
```
:::

::: {.cell .markdown id="xjrz_cApKa5v"}
Fungsi `store_many_disk` digunakan untuk menyimpan sebuah array gambar
beserta labelnya ke dalam disk dengan menyimpan setiap gambar dalam
format .png dan labelnya dalam sebuah file .csv terpisah.

Fungsi `store_many_lmdb` digunakan untuk menyimpan sebuah array gambar
beserta labelnya ke dalam basis data LMDB dengan menyimpan setiap gambar
dan labelnya dalam sebuah transaksi tunggal di dalam basis data
tersebut.

Fungsi `store_many_hdf5` digunakan untuk menyimpan sebuah array gambar
beserta labelnya ke dalam sebuah file HDF5 dengan menyimpan seluruh
array gambar dalam satu dataset bernama \"images\" dan seluruh array
label dalam satu dataset bernama \"meta\".

Ketiga fungsi tersebut menerima dua argumen: array gambar dengan dimensi
(N, 32, 32, 3) dan array label dengan dimensi (N, 1), di mana N adalah
jumlah gambar yang akan disimpan.
:::

::: {.cell .code id="v0bEeYbTKa5v" outputId="65461c6d-3dc1-40ed-ad0b-262a4dda6c46"}
``` python
cutoffs = [10, 100, 1000, 10000, 100000]

# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)

# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```

::: {.output .stream .stdout}
    (100000, 32, 32, 3)
    (100000,)
:::
:::

::: {.cell .markdown id="rcd3Xsd4Ka5w"}
Kode tersebut menggunakan fungsi np.concatenate untuk menggandakan array
images dan labels sehingga totalnya mencapai 100,000 gambar dan label.
Ini dilakukan dengan menggabungkan array images dan labels ke dalam
dirinya sendiri dua kali, menggunakan axis=0 yang berarti operasi
penggabungan dilakukan secara vertikal (along the first dimension).

-   Penggabungan Array: Fungsi np.concatenate digunakan untuk
    menggabungkan dua atau lebih array numpy. Dalam kasus ini, array
    images dan labels digabungkan dua kali, menghasilkan array yang
    berisi dua kali jumlah elemen asli. Fungsi ini memastikan bahwa
    array yang digabungkan memiliki bentuk yang sama, kecuali pada
    dimensi yang ditentukan oleh parameter axis.
-   Penggunaan axis=0: Parameter axis=0 dalam np.concatenate menentukan
    bahwa penggabungan dilakukan secara vertikal, yaitu, elemen-elemen
    dari array pertama ditambahkan ke array kedua secara vertikal. Ini
    berarti bahwa jumlah gambar dan label akan digandakan secara
    vertikal, menghasilkan total 100,000 gambar dan label 3.
-   Hasil: Setelah penggabungan, ukuran dari array images menjadi
    (100000, 32, 32, 3), yang menunjukkan bahwa ada 100,000 gambar,
    masing-masing dengan dimensi 32x32 piksel dan 3 kanal warna.
    Sementara itu, ukuran dari array labels menjadi (100000,), yang
    menunjukkan bahwa ada 100,000 label, masing-masing sesuai dengan
    gambar yang sesuai.
:::

::: {.cell .code id="HexQBiqTKa5w" outputId="27ec246e-9514-49fb-9995-bbb1b49fa378"}
``` python
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)

from timeit import timeit

store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.043001499999945736
    Method: lmdb, Time usage: 0.013635200000180703
    Method: hdf5, Time usage: 0.054849900000135676
    Method: disk, Time usage: 0.16676769999980934
    Method: lmdb, Time usage: 0.005678900000020803
    Method: hdf5, Time usage: 0.0023858000001837354
    Method: disk, Time usage: 1.6323519000000033
    Method: lmdb, Time usage: 0.03800269999987904
    Method: hdf5, Time usage: 0.005430300000170973
    Method: disk, Time usage: 12.146795200000042
    Method: lmdb, Time usage: 0.29728830000021844
    Method: hdf5, Time usage: 0.025982599999679223
    Method: disk, Time usage: 142.75358489999962
    Method: lmdb, Time usage: 4.296912700000121
    Method: hdf5, Time usage: 0.46554439999999886
:::
:::

::: {.cell .markdown id="gJ12CilBKa5w"}
Kode tersebut menggunakan modul timeit untuk mengukur waktu eksekusi
dari tiga metode penyimpanan gambar dalam jumlah yang berbeda: disk,
LMDB, dan HDF5. Ini dilakukan dengan menjalankan setiap fungsi
penyimpanan sekali untuk setiap nilai cutoff dalam list cutoffs, yang
mencakup jumlah gambar yang akan disimpan. Hasilnya disimpan dalam
dictionary store_many_timings untuk analisis lebih lanjut.

-   Metode Disk: Metode ini menyimpan gambar sebagai file .png dan label
    sebagai file .csv di disk. Waktu eksekusi yang dihasilkan bervariasi
    tergantung pada jumlah gambar yang disimpan. Misalnya, untuk 10
    gambar, waktu eksekusi adalah 0.043001499999945736 detik, untuk 100
    gambar adalah 0.16676769999980934 detik, dan untuk 100,000 gambar
    adalah 142.75358489999962 detik. Ini menunjukkan bahwa metode disk
    menjadi tidak efisien saat menangani jumlah gambar yang besar.
-   Metode LMDB: Metode ini menyimpan gambar dan label ke dalam database
    LMDB. LMDB menggunakan file yang dipetakan ke memori, memberikan
    performa I/O yang lebih baik dibandingkan dengan metode disk. Waktu
    eksekusi yang dihasilkan bervariasi tergantung pada jumlah gambar
    yang disimpan. Misalnya, untuk 10 gambar, waktu eksekusi adalah
    0.005678900000020803 detik, untuk 100 gambar adalah
    0.03800269999987904 detik, dan untuk 100,000 gambar adalah
    4.296912700000121 detik. Ini menunjukkan bahwa LMDB menawarkan
    performa yang lebih baik dibandingkan dengan metode disk, tetapi
    masih memiliki peningkatan waktu eksekusi seiring bertambahnya
    jumlah gambar.
-   Metode HDF5: Metode ini menyimpan gambar dan label ke dalam file
    HDF5. HDF5 dirancang untuk menyimpan data yang kompleks dan besar
    dalam format yang efisien dan mudah diakses. Waktu eksekusi yang
    dihasilkan bervariasi tergantung pada jumlah gambar yang disimpan.
    Misalnya, untuk 10 gambar, waktu eksekusi adalah
    0.0023858000001837354 detik, untuk 100 gambar adalah
    0.005430300000170973 detik, dan untuk 100,000 gambar adalah
    0.46554439999999886 detik. Ini menunjukkan bahwa HDF5 menawarkan
    performa yang paling baik dibandingkan dengan metode disk dan LMDB,
    dengan waktu eksekusi yang relatif konstan seiring bertambahnya
    jumlah gambar. Hasil ini menunjukkan bahwa LMDB menawarkan performa
    yang lebih baik dibandingkan dengan metode disk, tetapi masih
    memiliki peningkatan waktu eksekusi seiring bertambahnya jumlah
    gambar. Sementara itu, HDF5 menawarkan performa yang paling baik,
    dengan waktu eksekusi yang relatif konstan seiring bertambahnya
    jumlah gambar. Pilihan metode penyimpanan tergantung pada kebutuhan
    spesifik aplikasi, termasuk ukuran data, kebutuhan akses, dan
    preferensi pengguna terhadap format file.
:::

::: {.cell .code id="xDrE7KqXKa5w" outputId="80088ebc-19c3-4754-e034-557f399099aa"}
``` python
import matplotlib.pyplot as plt

def plot_with_legend(
    x_range, y_data, legend_labels, x_label, y_label, title, log=False
):
    """ Displays a single plot with multiple datasets and matching legends.
        Parameters:
        --------------
        x_range         list of lists containing x data
        y_data          list of lists containing y values
        legend_labels   list of string legend labels
        x_label         x axis label
        y_label         y axis label
    """
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 7))

    if len(y_data) != len(legend_labels):
        raise TypeError(
            "Error: number of data sets does not match number of labels."
        )

    all_plots = []
    for data, label in zip(y_data, legend_labels):
        if log:
            temp, = plt.loglog(x_range, data, label=label)
        else:
            temp, = plt.plot(x_range, data, label=label)
        all_plots.append(temp)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=all_plots)
    plt.show()

# Getting the store timings data to display
disk_x = store_many_timings["disk"]
lmdb_x = store_many_timings["lmdb"]
hdf5_x = store_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Storage time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Log storage time",
    log=True,
)
```

::: {.output .stream .stderr}
    C:\Users\ASUS\AppData\Local\Temp\ipykernel_12376\2568719458.py:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_970b7e57034d4a98828981e39955f5a7/7162a3c384d7ee9787d100df609eed34319f5bf5.png)
:::

::: {.output .stream .stderr}
    C:\Users\ASUS\AppData\Local\Temp\ipykernel_12376\2568719458.py:15: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
:::

::: {.output .display_data}
![](vertopal_970b7e57034d4a98828981e39955f5a7/c39ac9c3a10747bb7a215a637405fcb7a1a7e494.png)
:::
:::

::: {.cell .code id="RCSfDmrBKa5w"}
``` python
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))

    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])

    return image, label
```
:::

::: {.cell .markdown id="Xnx8_uvrKa5w"}
Fungsi read_single_disk yang didefinisikan dalam kode ini bertujuan
untuk membaca satu gambar dan labelnya dari disk. Fungsi ini
mengasumsikan bahwa gambar disimpan sebagai file .png dan label disimpan
sebagai file .csv dalam direktori yang ditentukan oleh disk_dir.

-   Parameter Fungsi: Fungsi ini menerima satu parameter: image_id, yang
    merupakan ID unik integer untuk gambar.
-   Membaca Gambar: Fungsi ini menggunakan modul Image dari library PIL
    (Python Imaging Library) untuk membuka file gambar dengan nama yang
    sesuai dengan image_id dan mengubahnya menjadi array numpy. Ini
    memungkinkan gambar untuk diakses dan diproses sebagai array, yang
    sangat berguna untuk operasi pengolahan gambar atau pembelajaran
    mesin.
-   Membaca Label: Fungsi ini membuka file CSV yang sesuai dengan
    image_id dan membaca label dari file tersebut. Label diasumsikan
    berada di baris pertama file CSV dan diubah menjadi integer. Fungsi
    ini menggunakan modul csv untuk membaca file CSV, dengan delimiter
    spasi dan karakter kutip \|.
-   Pengembalian Nilai: Fungsi ini mengembalikan dua nilai: array numpy
    yang mewakili gambar dan label yang sesuai dengan gambar tersebut.
:::

::: {.cell .code id="VAzxlxbmKa5x"}
``` python
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()

    return image, label
```
:::

::: {.cell .markdown id="U_hLy6rrKa5x"}
Fungsi read_single_lmdb yang didefinisikan dalam kode ini bertujuan
untuk membaca satu gambar dan labelnya dari database LMDB. LMDB adalah
database key-value yang dirancang untuk penggunaan dalam memori dan
disk, menawarkan akses yang cepat dan efisien terhadap data.

-   Parameter Fungsi: Fungsi ini menerima satu parameter: image_id, yang
    merupakan ID unik integer untuk gambar.
-   Membuka Environment LMDB: Fungsi ini membuka environment LMDB yang
    berada di direktori yang ditentukan oleh lmdb_dir dengan nama
    single_lmdb. Environment ini dibuka dalam mode hanya-baca
    (readonly=True), yang berarti tidak ada operasi penulisan yang
    diizinkan.
-   Transaksi Pembacaan: Fungsi ini membuka transaksi pembacaan baru ke
    dalam environment LMDB. Dalam transaksi ini, gambar dan label
    diambil dari database. Kunci untuk mengambil gambar dan label dibuat
    dengan menggunakan image_id yang diubah menjadi string dengan format
    8 digit (misalnya, \"00000001\" untuk image_id 1). Kunci ini
    kemudian di-encode menjadi bytes dengan kode ASCII.
-   Membaca Data: Data yang diambil dari database di-decode dan
    di-pickle untuk mengubahnya kembali menjadi objek CIFAR_Image (yang
    diasumsikan telah didefinisikan sebelumnya). Objek ini kemudian
    digunakan untuk mengambil gambar dan label yang sesuai.
-   Penutupan Environment: Setelah gambar dan label diambil, environment
    LMDB ditutup.
-   Pengembalian Nilai: Fungsi ini mengembalikan dua nilai: array numpy
    yang mewakili gambar dan label yang sesuai dengan gambar tersebut.
:::

::: {.cell .code id="fmGnOWnLKa5x"}
``` python
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))

    return image, label
```
:::

::: {.cell .markdown id="wjHKK0vBKa5x"}
Fungsi `read_single_hdf5` digunakan untuk membaca sebuah gambar dan
label yang tersimpan dalam file HDF5. Fungsi ini menerima satu argumen
yaitu `image_id`, yang merupakan ID unik untuk gambar yang akan dibaca.

Pertama, file HDF5 dibuka dengan mode baca dan tulis (r+ mode)
menggunakan `h5py.File`. Kemudian, gambar dibaca dari dataset \"image\"
dalam file HDF5 dan diubah menjadi array numpy dengan tipe data `uint8`
(unsigned integer 8-bit). Selanjutnya, label dibaca dari dataset
\"meta\" dan juga diubah menjadi integer dengan tipe data `uint8`.

Setelah membaca data gambar dan label, file HDF5 ditutup dan gambar
beserta labelnya dikembalikan sebagai output dari fungsi.

Dengan fungsi ini, Anda dapat dengan mudah membaca gambar dan label dari
file HDF5 berdasarkan ID unik gambar yang diberikan.
:::

::: {.cell .code id="Cm1sY-xtKa5x"}
``` python
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```
:::

::: {.cell .markdown id="XoSqaGVeKa5y"}
Kode `_read_single_funcs` membuat sebuah kamus yang berisi tiga fungsi:
`read_single_disk`, `read_single_lmdb`, dan `read_single_hdf5`,
masing-masing terkait dengan metode membaca data dari lokasi penyimpanan
yang berbeda (`disk`, `lmdb`, dan `hdf5`). Hal ini memungkinkan
penggunaan kamus ini untuk memilih metode membaca yang sesuai dengan
kebutuhan aplikasi, dengan cukup memanggil fungsi yang sesuai dengan
kunci yang diinginkan (seperti `'disk'` untuk membaca dari disk,
`'lmdb'` untuk membaca dari LMDB, dan `'hdf5'` untuk membaca dari HDF5).
Dengan cara ini, fleksibilitas dalam memilih cara membaca data dapat
diterapkan dengan mudah dalam program Anda.
:::

::: {.cell .code id="CrGdor9yKa5y" outputId="6a67ebd3-8fca-443b-9069-57af90d460d9"}
``` python
from timeit import timeit

read_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

::: {.output .stream .stdout}
    Method: disk, Time usage: 0.03221690000009403
    Method: lmdb, Time usage: 0.029903999999987718
    Method: hdf5, Time usage: 0.020119400000112364
:::
:::

::: {.cell .markdown id="e9ghRga9Ka5y"}
Kode tersebut menggunakan modul timeit untuk mengukur waktu eksekusi
dari tiga metode pembacaan gambar: disk, LMDB, dan HDF5. Ini dilakukan
dengan menjalankan setiap fungsi pembacaan sekali untuk gambar dengan ID
0. Hasilnya disimpan dalam dictionary read_single_timings untuk analisis
lebih lanjut.

-   Metode Disk: Waktu eksekusi untuk membaca satu gambar dari disk
    adalah 0.03221690000009403 detik. Ini menunjukkan bahwa metode disk
    cukup cepat untuk operasi pembacaan tunggal, meskipun mungkin tidak
    secepat metode lainnya untuk operasi dalam skala yang lebih besar.
-   Metode LMDB: Waktu eksekusi untuk membaca satu gambar dari database
    LMDB adalah 0.029903999999987718 detik. LMDB dirancang untuk akses
    yang cepat ke data dalam memori dan disk, sehingga performa ini
    menunjukkan bahwa metode ini efisien untuk operasi pembacaan
    tunggal.
-   Metode HDF5: Waktu eksekusi untuk membaca satu gambar dari file HDF5
    adalah 0.020119400000112364 detik. HDF5 dirancang untuk menyimpan
    data yang kompleks dan besar dalam format yang efisien dan mudah
    diakses, sehingga performa ini menunjukkan bahwa metode ini sangat
    efisien untuk operasi pembacaan tunggal. Hasil ini menunjukkan bahwa
    semua metode menawarkan performa yang cukup baik untuk operasi
    pembacaan tunggal gambar. Namun, HDF5 menunjukkan performa yang
    paling baik, dengan waktu eksekusi yang paling rendah. Ini
    menunjukkan bahwa HDF5 mungkin merupakan pilihan yang baik untuk
    aplikasi yang memerlukan akses cepat dan efisien ke data gambar dan
    label, terutama jika data tersebut besar dan kompleks
:::

::: {.cell .code id="P7daJmvHKa5y"}
``` python
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))

    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels

def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels

def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)
```
:::

::: {.cell .markdown id="ht0QHtw5Ka5y"}
Kode di atas mendefinisikan tiga fungsi, `read_many_disk`,
`read_many_lmdb`, dan `read_many_hdf5`, untuk membaca sejumlah gambar
dan label dari berbagai lokasi penyimpanan (disk, LMDB, HDF5).

Fungsi `read_many_disk` membaca gambar dan label dari disk dengan
membuka file .png untuk setiap gambar dan file .csv untuk labelnya.
Proses pembacaan dilakukan dengan loop dan menggunakan modul PIL untuk
membaca gambar.

Fungsi `read_many_lmdb` membaca gambar dan label dari basis data LMDB
dengan membuka transaksi baca menggunakan modul lmdb. Proses pembacaan
dilakukan dengan loop untuk mengambil data gambar dan label dari setiap
kunci yang sesuai dalam basis data.

Fungsi `read_many_hdf5` membaca gambar dan label dari file HDF5 dengan
membuka file HDF5 dan membaca dataset \"images\" dan \"meta\" dari file
tersebut.

Kamus `_read_many_funcs` digunakan untuk mengelompokkan ketiga fungsi
pembacaan tersebut berdasarkan lokasi penyimpanan yang berbeda (disk,
LMDB, HDF5).

Kode ini memungkinkan untuk membaca sejumlah gambar dan label dari
berbagai lokasi penyimpanan dengan mudah dan fleksibel, tergantung pada
kebutuhan aplikasi.
:::

::: {.cell .code id="skQl2FFrKa5z" outputId="7269608a-3cd3-40b6-bc92-b2585c7a4c03"}
``` python
from timeit import timeit

read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")
```

::: {.output .error ename="IndexError" evalue="list index out of range"}
    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)
    Cell In [47], line 7
          5 for cutoff in cutoffs:
          6     for method in ("disk", "lmdb", "hdf5"):
    ----> 7         t = timeit(
          8             "_read_many_funcs[method](num_images)",
          9             setup="num_images=cutoff",
         10             number=1,
         11             globals=globals(),
         12         )
         13         read_many_timings[method].append(t)
         15         # Print out the method, cutoff, and elapsed time

    File c:\Users\ASUS\AppData\Local\Programs\Python\Python39\lib\timeit.py:233, in timeit(stmt, setup, timer, number, globals)
        230 def timeit(stmt="pass", setup="pass", timer=default_timer,
        231            number=default_number, globals=None):
        232     """Convenience function to create Timer object and call timeit method."""
    --> 233     return Timer(stmt, setup, timer, globals).timeit(number)

    File c:\Users\ASUS\AppData\Local\Programs\Python\Python39\lib\timeit.py:177, in Timer.timeit(self, number)
        175 gc.disable()
        176 try:
    --> 177     timing = self.inner(it, self.timer)
        178 finally:
        179     if gcold:

    File <timeit-src>:6, in inner(_it, _timer)

    Cell In [42], line 23, in read_many_disk(num_images)
         19     reader = csv.reader(
         20         csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
         21     )
         22     for row in reader:
    ---> 23         labels.append(int(row[0]))
         24 return images, labels

    IndexError: list index out of range
:::
:::

::: {.cell .markdown id="EWegskL9Ka5z"}
Kode di atas mengukur waktu yang dibutuhkan untuk membaca sejumlah
gambar dari berbagai metode penyimpanan (disk, LMDB, HDF5) dengan
menggunakan `timeit`. Hasil pengukuran waktu disimpan dalam kamus
`read_many_timings`.

Pada setiap iterasi, kode melakukan pengukuran waktu untuk membaca
sejumlah gambar (`num_images`) berdasarkan metode penyimpanan yang
ditentukan dalam variabel `method`. Pengukuran waktu dilakukan sekali
saja (number=1) untuk setiap metode.

Hasil pengukuran waktu kemudian disimpan dalam kamus `read_many_timings`
untuk masing-masing metode penyimpanan.

Selain itu, kode juga mencetak informasi tentang metode penyimpanan,
jumlah gambar yang dibaca, dan waktu yang dibutuhkan untuk pembacaan
tersebut.

Kode ini memberikan informasi yang berguna untuk mengevaluasi performa
relatif dari berbagai metode pembacaan data tergantung pada jumlah
gambar yang dibaca (`cutoff`) dari setiap metode penyimpanan.
:::
