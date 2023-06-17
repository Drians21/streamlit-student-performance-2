import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

# st.sidebar.success("Silahkan Pilih Halaman Di atas 🔺")

st.title(':red[Pertanyaan]')
st.write('---')
df = pd.read_csv('data/StudentsPerformance.csv')

#Rename Columns
df.columns = df.columns.str.replace(" ", "_").copy()
df = df.rename({'parental_level_of_education':'parent_education', 
                    'race/ethnicity':'ethnicity', 
                    'test_preparation_course':'prep_course'}, 
                    axis=1).copy()

#Menambah Column average_score
df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)


# Define independent variables (factors) and target variable (performance_level)
X = df[['gender','parent_education', 'lunch', 'prep_course']]
y = df['average_score']

# Convert target variable to categorical
y = pd.cut(y, bins=[0, 40, 60, 100], labels=['Low', 'Medium', 'High'])

# Convert categorical variables to dummy variables (one-hot encoding)
X = pd.get_dummies(X)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict performance levels for the test data
y_pred = model.predict(X_test)



#Pertanyaan

choice = st.selectbox(
    '#### :green[Pilih Pertanyaan :] 🔻',
    [
    '1','2','3','4','5','6','7','8'
    ])

st.write('')
st.write('')

if choice == '1':
    st.warning('##### :red[1. Apakah ada korelasi antara latar belakang orang tua dengan kinerja siswa?]')
    education_mapping = {
    'some high school': 1,
    'high school': 2,
    'some college': 3,
    'associate\'s degree': 4,
    'bachelor\'s degree': 5,
    'master\'s degree': 6
    }

    data_parent = df['parent_education'].map(education_mapping)

    # Menghitung korelasi Pearson
    pearson_corr = data_parent.corr(df['average_score'], method='pearson')

    # Menghitung korelasi Spearman
    spearman_corr = data_parent.corr(df['average_score'], method='spearman')

    st.code("""

            education_mapping = {
            'some high school': 1,
            'high school': 2,
            'some college': 3,
            'associate\'s degree': 4,
            'bachelor\'s degree': 5,
            'master\'s degree': 6
            }

            data_parent = df['parent_education'].map(education_mapping)
            pearson_corr = data_parent.corr(df['average_score'], method='pearson')
            spearman_corr = data_parent.corr(df['average_score'], method='spearman')                                
    """)

    st.success(f":red[Korelasi Pearson] : {pearson_corr}")
    st.success(f":red[Korelasi Spearman] : {spearman_corr}" ) 


    # Membuat urutan kategori tingkat pendidikan
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

    # Menghitung rata-rata skor berdasarkan tingkat pendidikan orang tua
    parent_edu_performance = df.groupby(data_parent)['average_score'].mean()
    # Plot bar dengan urutan tingkat pendidikan yang diinginkan
    plt.figure(figsize=(10, 6))
    sns.barplot(x=data_parent, y=df['average_score'], data=df, order=education_order.categories)
    plt.xlabel('Parent Education Level')
    plt.ylabel('Average Score')
    plt.title('Parent Education Level vs. Average Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

elif choice == '2':  
    st.warning('##### :red[2. Apakah persiapan ujian mempengaruhi kinerja siswa dalam ujian?]')
    prep_mapping = {
        'completed': 1,
        'none': 0,
    }

    data_prep = df['prep_course'].map(prep_mapping)

    # Menghitung korelasi Pearson dan Spearman
    pearson_corr = data_prep.corr(df['average_score'], method='pearson')
    spearman_corr = data_prep.corr(df['average_score'], method='spearman')  

    st.code("""

            prep_mapping = {
            'completed': 1,
            'none': 0,
            }

            data_prep = df['prep_course'].map(education_mapping)
            pearson_corr = data_prep.corr(df['average_score'], method='pearson')
            spearman_corr = data_prep.corr(df['average_score'], method='spearman')      
    """)

    st.success(f':red[pearson_corr] : {pearson_corr}')      
    st.success(f':red[spearman_corr] : {spearman_corr}')   

    # Grafik
    prep_course_performance = df.groupby('prep_course')['average_score'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['prep_course'], y=df['average_score'], data=df)
    plt.xlabel('Course Preparation')
    plt.ylabel('Average Score')
    plt.title('Course Preparation vs. Average Score')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

elif choice == '3':
    st.warning('##### :red[3. Faktor-faktor apa saja mempengaruhi dan paling penting untuk kinerja siswa dalam ujian?]')
    # features = df.drop(columns=['writing_score', 'math_score', 'reading_score', 'average_score']).columns
    # features = df[['gender','parent_education', 'lunch', 'prep_course']]
    feature_names = X.columns.tolist()
    # features = pd.DataFrame(df.data, columns=df.feature_names)

    feat_img_fig = plt.figure(figsize=(6,4))
    ax1 = feat_img_fig.add_subplot(111)
    skplt.estimators.plot_feature_importances(model, feature_names=feature_names, ax=ax1, x_tick_rotation=90)
    st.pyplot(feat_img_fig, use_container_width=True)

    # factors = ['gender', 'ethnicity', 'parent_education', 'lunch', 'prep_course']
    # for factor in factors:
    #     factor_performance = df.groupby(factor)['average_score'].mean()
    #     plt.bar(factor_performance.index, factor_performance.values)
    #     plt.xlabel(factor.capitalize())
    #     plt.ylabel('Average Score')
    #     plt.title(factor.capitalize() + ' vs. Average Score')
    #     plt.xticks(rotation=90)
    #     st.pyplot(plt)

elif choice == '4':                                                
# 4. Bagaimana distribusi nilai siswa berdasarkan latar belakang orang tua dan persiapan ujian?
    st.warning('##### :red[4. Bagaimana distribusi nilai siswa berdasarkan latar belakang orang tua dan persiapan ujian?]')

    # Membuat urutan kategori tingkat pendidikan
    education_order = CategoricalDtype(categories=[
        'some high school',
        'high school',
        'some college',
        'associate\'s degree',
        'bachelor\'s degree',
        'master\'s degree'
    ], ordered=True)

    # Mengkonversi kolom 'parent_education' menjadi tipe kategori dengan urutan yang ditentukan
    data_parent2 = df['parent_education'].astype(education_order)
    # Membuat plot violin
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=data_parent2, y='average_score', hue='prep_course', data=df)
    plt.xlabel('Parent Education')
    plt.ylabel('Average Score')
    plt.title('Distribution of Student Scores by Parent Education and Prep Course')
    # plt.xticks(rotation=45)
    plt.legend(title='Prep Course')
    plt.tight_layout()
    st.pyplot(plt)

elif choice == '5':  
    # 5. Apakah terdapat hubungan antara kinerja siswa dalam mata pelajaran tertentu dengan kinerja mereka dalam mata pelajaran lainnya?
    st.warning('##### :red[5. Apakah terdapat hubungan antara kinerja siswa dalam mata pelajaran tertentu dengan kinerja mereka dalam mata pelajaran lainnya?]')
    plt.figure(figsize=(8,5))
    sns.heatmap(pd.get_dummies(df[['writing_score','reading_score','math_score']],drop_first=True).corr(),cmap='viridis',annot=True)
    plt.title('Correlation between features')
    st.pyplot(plt)
    # st.write('')

elif choice == '6': 
    # 6. Apakah ada korelasi antara gender dan nilai rata-rata siswa?

    st.warning('##### :red[6. Apakah ada korelasi antara gender dan nilai rata-rata siswa?]')

    # Mengubah latar belakang pendidikan orang tua menjadi nilai numerik
    gender_mapping = {
    'male': 0,
    'female': 1,
    }

    data_gender = df['gender'].map(gender_mapping)

    # Menghitung korelasi Pearson
    pearson_corr = data_gender.corr(df['average_score'], method='pearson')

    # Menghitung korelasi Spearman
    spearman_corr = data_gender.corr(df['average_score'], method='spearman')

    st.code("""
        
            gender_mapping = {
            'male': 0,
            'female': 1,
            }

            data_gender = df['gender'].map(education_mapping)
            pearson_corr = data_gender.corr(df['average_score'], method='pearson')
            spearman_corr = data_gender.corr(df['average_score'], method='spearman')                            
    """)

    st.success(f":red[Korelasi Pearson] : {pearson_corr}")
    st.success(f":red[Korelasi Spearman] : {spearman_corr}" ) 

    # Menghitung rata-rata skor berdasarkan tingkat pendidikan orang tua
    gender_graf = df.groupby(data_gender)['average_score'].mean()
    # Plot bar dengan urutan tingkat pendidikan yang diinginkan
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['gender'], y=df['average_score'], data=df)
    plt.xlabel('Gender')
    plt.ylabel('Average Score')
    plt.title('Gender vs. Average Score')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

elif choice == '7': 
    # 7. Apakah ada korelasi antara lunch dengan kinerja nilai siswa?
    st.write('')
    st.warning('##### :red[7. Apakah ada korelasi antara lunch dengan kinerja nilai siswa?]')

    # Mengubah latar belakang pendidikan orang tua menjadi nilai numerik
    lunch_mapping = {
        'standard': 1,
        'free/reduced': 0,
        }

    data_lunch = df['lunch'].map(lunch_mapping)

    # Menghitung korelasi Pearson
    pearson_corr = data_lunch.corr(df['average_score'], method='pearson')
    spearman_corr = data_lunch.corr(df['average_score'], method='spearman')

    st.code("""
        
        lunch_mapping = {
        'standard': 1,
        'free/reduced': 0,
        }

        data_lunch = df['lunch'].map(lunch_mapping)

        # Menghitung korelasi Pearson
        pearson_corr = data_lunch.corr(df['average_score'], method='pearson')
        spearman_corr = data_lunch.corr(df['average_score'], method='spearman')                            
    """)

    st.success(f":red[Korelasi Pearson] : {pearson_corr}")
    st.success(f":red[Korelasi Spearman] : {spearman_corr}" )

    # Plot bar dengan urutan tingkat pendidikan yang diinginkan
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['lunch'], y=df['average_score'], data=df)
    plt.xlabel('Lunch')
    plt.ylabel('Average Score')
    plt.title('Gender vs. Average Score')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)  

elif choice == '8': 
    # 8. Apakah kita bisa membuat model dan memprediksi dari data ini?
    st.warning('##### :red[8. Apakah kita bisa membuat model dan memprediksi dari data ini?]')

    st.code("""

            # Define independent variables (factors) and target variable (performance_level)

            X = df[['gender','parent_education', 'lunch', 'prep_course']]
            y = df['average_score']

            # Convert target variable to categorical

            y = pd.cut(y, bins=[0, 40, 60, 100], labels=['Low', 'Medium', 'High'])

            # Convert categorical variables to dummy variables (one-hot encoding)

            X = pd.get_dummies(X)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

            # Build RandomForestClassifier model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Predict performance levels for the test data
            y_pred = model.predict(X_test)
    """)

    accuracy = accuracy_score(y_test, y_pred)
    st.success(f'Nilai Akurasi Model : {accuracy}')
    st.write('')
    st.write('Confusion Matrix')
    conf_mat_fig = plt.figure(figsize=(5,3))
    ax1 = conf_mat_fig.add_subplot(111)
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax=ax1)
    st.pyplot(conf_mat_fig, use_container_width=True)
    st.write('')


    st.write('Classification Report')
    st.code(classification_report(y_test,y_pred))