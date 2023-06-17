import streamlit as st
import pandas as pd

# Dataset
st.title(':green[Dataset]')
st.write('---')

st.write(':green[Student Perfomance in Exams]. Kumpulan data ini terdiri dari nilai yang diperoleh siswa dalam berbagai mata pelajaran. Data ini bisa digunakan untuk memahami pengaruh latar belakang orang tua, persiapan ujian dll terhadap kinerja siswa.')
st.write(':red[Referensi Dataset :] https://www.kaggle.com/datasets/spscientist/students-performance-in-exams')
df = pd.read_csv('data/StudentsPerformance.csv')
st.write(df)

st.write('')
st.write('#### :green[Informasi Dataset]')
st.image('images/informasi-data.png')
st.write('Kami menggunakan data tersebut dengan :blue[Jumlah baris 1000] dan :blue[Jumlah Kolom 9].')