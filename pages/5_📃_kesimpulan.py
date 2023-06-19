import streamlit as st

st.title(':green[Kesimpulan]')
st.write('---')

st.write("""
#### :blue[Korelasi Positif]
Korelasi antara variabel independen dengan dependen semuanya bernilai positif yang artinya semakin baik variabel dependen maka semakin baik juga variabel independennya.

#### :blue[Hasil klasifikasi]
Dalam model ini kami melakukan klasifikasi karena data tersebut memiliki kategori yang sesuai dengan klasifikasi salah satunya datanya lebih ke categorical. Sehinggan prediksi tidak bisa dilakukan untuk data ini.

Algoritma yang kami gunakan yaitu, Random Forest. Dengan Random Forest, Akurasi klasifikasi yang kami dapatkan yaitu 0.64 / 64% dengan feature importance paling tinggi adalah lunch, kemudian prep_test lalu diikuti variabel lainnya. Maka dari itu untuk lunch harus dan prep_test harus diperhatikan karena banyak mempengaruhi hasil prediksi.

:blue[Dengan demikian], Makanan yang tepat dan persiapan yang matang berdampak paling besar pada hasil ujian. Penting untuk menjaga pola makan baik dan meluangkan waktu yang cukup untuk mempersiapkan diri agar mencapai tingkat intelegensi yang optimal. Dengan perhatian pada faktor-faktor ini, kita dapat mencapai kesuksesan dalam ujian.
""")
