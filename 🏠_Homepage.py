import streamlit as st
# import pandas as pd

st.set_page_config(
    page_title="Student Performance",
    page_icon="📋"
)

# st.sidebar.success("Silahkan Pilih Halaman Di atas 🔺")

st.markdown("""
    ### :red[Kisah Intelektualitas:] Menjelajahi Faktor-Faktor Penggerak intelegensi Pelajar di Indonesia :bar_chart:
""")
st.write('---------')

st.image('images/PISSA-2018.png', caption='Sumber:https://gurudikdas.kemdikbud.go.id/storage/users/3/Berita/PISSA%202018.png', use_column_width='always')
st.write(':blue[PISA (Program for International Student Assessment)] atau program penilaian pelajar internasional yang diselenggarakan setiap 3 tahun guna menguji para pelajar yang berusia 15 tahun untuk mengukur :blue[intelengsi] mereka.')
st.write(':blue[Indonesia] mengalami :blue[peningkatan] nilai PISA sejak 2000, :blue[terutama di bidang matematika]. :blue[Namun, pada PISA 2018,] skor Indonesia menurun di semua bidang. Penurunan paling tajam terjadi dalam membaca.')
st.write(':red[Kenapa hal itu bisa terjadi?]')
st.write('Padahal kalau dibaratkan sebenarnya kita berada dalam ombak yang sama. Namun yang menjadi masalah adalah kondisinya yang berdeda. misalnya, ada yang menggunakan kapal besar, perahu, bahkan perahu kecil. Mereka yang berada di perahu kecil ini membutuhkan usaha yang ekstra. Bahkan mereka kekurangan instrumen untuk menunjang hal itu. Salah satunya adalah nutrisi/makanan yang mereka makan. Mereka mereka yang berada di perahu kecil, mereka - mereka yang dalam kondisi ektrim akan memakan makanan seadanya sehingga nutrisi mereka tidak terpenuhi untuk menunjang proses belajar itu.')
st.write('Dengan demikian kami tertaik untuk melakukan analisis untuk melihat faktor - faktor apa yang mempengaruhi hal itu. Untuk menganalisis kami mengambil dataset dari kaggle.')
