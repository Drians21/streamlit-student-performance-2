import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from pandas.api.types import CategoricalDtype

# st.sidebar.success("Silahkan Pilih Halaman Di atas 🔺")

st.title(':red[EDA]')
st.write('---')

df = pd.read_csv('data/StudentsPerformance.csv')


#Rename Columns
st.warning('##### Rename Columns')

df.columns = df.columns.str.replace(" ", "_").copy()
df = df.rename({'parental_level_of_education':'parent_education', 
                    'race/ethnicity':'ethnicity', 
                    'test_preparation_course':'prep_course'}, 
                    axis=1).copy()

#Tampilan Pada Web
st.code("""
    
        df.columns = df.columns.str.replace(" ", "_").copy()
        df = df.rename({'parental_level_of_education':'parent_education', 
                        'race/ethnicity':'ethnicity', 
                        'test_preparation_course':'prep_course'}, 
                        axis=1).copy()
""")
st.write(df)


#Menambah Column average_score
st.write('')
st.warning('##### Menambah Kolom Average')

df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
st.code("""

        df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
""")
st.write(df)


st.write('')
st.write('')
choice = st.selectbox('#### :green[SUB MENU] 🔻',['Test Preparation', 'Gender', 'Ethnicity', 'Lunch', 'Parent Education', 'Korelasi'])

if choice == 'Test Preparation':
    # st.warning('##### Test Prepareation')
    test_preparation = df.groupby('prep_course')['prep_course'].count()

    ordered_categories = sorted(df['prep_course'].unique())
    plt.figure(figsize=(10,6))
    sns.countplot(data=df,x='prep_course',palette='deep', order=ordered_categories)
    for i, count in enumerate(test_preparation[ordered_categories]):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.title('Test Preparation')
    st.pyplot(plt)

elif choice == 'Gender':
    # st.write('')
    # st.warning('##### Gender')
    gender = df.groupby('gender')['gender'].count()

    ordered_categories = sorted(df['gender'].unique())
    plt.figure(figsize=(10,6))
    sns.countplot(data=df,x='gender',palette='deep', order=ordered_categories)
    for i, count in enumerate(gender[ordered_categories]):
        plt.text(i, count, str(count), ha='center', va='bottom')

    # plt.figure(figsize=(4,4))
    # sns.countplot(data=df,x='gender',palette='deep')
    plt.title('Gender')
    st.pyplot(plt)

elif choice == 'Ethnicity':
    # st.write('')
    # st.warning('##### Ethnicity')
    ethnicity = df.groupby('ethnicity')['ethnicity'].count()

    ordered_categories = sorted(df['ethnicity'].unique())
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='ethnicity', palette='deep', order=ordered_categories)
    for i, count in enumerate(ethnicity[ordered_categories]):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.title('Ethnicity')
    plt.show()
    st.pyplot(plt)

elif choice == 'Lunch':
    # st.write('')
    # st.warning('##### Lunch')
    lunch = df.groupby('lunch')['lunch'].count()

    ordered_categories = sorted(df['lunch'].unique())
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='lunch', palette='deep', order=ordered_categories)
    for i, count in enumerate(lunch[ordered_categories]):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.title('Lunch')
    st.pyplot(plt)

elif choice == 'Parent Education':
    # st.write('')
    # st.warning('##### Parent Education')
    parent_education = df.groupby('parent_education')['parent_education'].count()
    st.write(parent_education)

    education_order = CategoricalDtype(categories=[
        'some high school',
        'high school',
        'some college',
        'associate\'s degree',
        'bachelor\'s degree',
        'master\'s degree'
    ], ordered=True)

    # Mengkonversi kolom 'parent_education' menjadi tipe kategori dengan urutan yang ditentukan
    data_parent = df['parent_education'].astype(education_order)
    plt.figure(figsize=(10,6))
    sns.countplot(data=df,x=data_parent ,palette='deep')
    plt.title('Parent Education')
    plt.xticks(rotation=90)
    st.pyplot(plt)

else:
    # st.write('')
    # st.warning('Korelasi Antara Variable')
    education_order = CategoricalDtype(categories=[
        'some high school',
        'high school',
        'some college',
        'associate\'s degree',
        'bachelor\'s degree',
        'master\'s degree'
    ], ordered=True)

    # Mengkonversi kolom 'parent_education' menjadi tipe kategori dengan urutan yang ditentukan
    data_parent = df['parent_education'].astype(education_order)
    
    education_order = {
        'some high school' : 0,
        'high school' : 1,
        'some college' : 2,
        'associate\'s degree' : 3,
        'bachelor\'s degree' : 4,
        'master\'s degree' : 5
    }

    prep_mapping = {
            'completed': 1,
            'none': 0,
        }

    gender_mapping = {
        'male': 0,
        'female': 1,
        }

    lunch_mapping = {
        'standard': 1,
        'free/reduced': 0,
        }

    ethnicity_mapping = {
        'group A' : 0,
        'group B' : 1,
        'group C' : 3,
        'group D' : 4,
        'group E' : 5
    }

    data_parent = data_parent.map(education_order)
    data_prep = df['prep_course'].map(prep_mapping)
    data_gender = df['gender'].map(gender_mapping)
    data_lunch = df['lunch'].map(lunch_mapping)
    data_ethnicity = df['ethnicity'].map(ethnicity_mapping)
    # data_ethnicity
        

    # Gabungkan kolom-kolom kategorikal menjadi satu DataFrame
    combined_df = pd.concat([data_lunch, data_prep, data_gender, data_parent, data_ethnicity, df['average_score']], axis=1)

    # Hitung korelasi menggunakan metode Pearson
    correlation_matrix = combined_df.corr(method='pearson')

    # Plot heatmap korelasi
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
    plt.title('Categorical Correlation Heatmap')
    st.pyplot(plt)



