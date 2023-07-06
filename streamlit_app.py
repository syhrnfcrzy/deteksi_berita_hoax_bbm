import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from nltk.corpus import stopwords
import time
import nltk
nltk.download('stopwords')
nltk.download('punkt')

st.set_page_config(page_icon="📰", page_title="Bahan Bakar Minyak", initial_sidebar_state="auto")

hide_menu_style = """
        <style>
        footer {visibility: visible;}
        footer:after{content:'Copyright @ 2023 Nabila Rizqi Amalia'; display:block; position:relative; color:white}
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

#background-image: url("https://www.settimolink.it/wp-content/uploads/2018/02/fakenews.jpg");
page_bg_img = f"""
<style>


[data-testid="stAppViewContainer"] > .main {{
background-size: 100%;
background-height: auto;
background-position: center;
background-color: #000;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

m = """
<style>

.stAlert {
    text-color: #fff;
}
div.stButton > button:first-child {
    background-color: #c20202;
    border-radius:20px 20px 20px 20px;
}
div.stButton > button:hover {
    background-color: #f30202;
    color: #fff;
    }
div.css-1om1ktf.e1y61itm0 {
        
        }
        textarea.st-cl {
          height: 150px;
          background-color: #fff;
          color: #000;
          font-family:"Roboto", serif;
          font-size: 15px;
        }
div.css-1dj3z61.e1iq63gx0 {
color: #000;
}
p {
color: #fff;
}
</style>"""
st.markdown(m, unsafe_allow_html=True)

st.title(":red[Deteksi Berita Hoax Bahan Bakar Minyak (BBM) Menggunakan Metode K-Nearst Neighbor]")

# Baca dataset berita
data = pd.read_csv('dataset berita hoax dan bukan hoax.csv')

data['Sumber'] = data['Sumber'].replace('https://www.kominfo.go.id', 'KOMINFO')
data['Sumber'] = data['Sumber'].replace('https://www.google.com', 'KOMPAS')

# Membuang kolom yang tidak diperlukan
data.drop('Unnamed: 0', axis=1, inplace=True)
data.drop('Teks', axis=1, inplace=True)
data.drop('Rangkuman', axis=1, inplace=True)
data.drop('Penulis', axis=1, inplace=True)

data = data.groupby('Judul').first().reset_index()

# Menghapus kata "[DISINFORMASI]" dan "[HOAKS]" pada kolom "Judul"
data['Judul'] = data['Judul'].str.replace('\[DISINFORMASI\]', '')
data['Judul'] = data['Judul'].str.replace('\[HOAKS\]', '')
data['Judul'] = data['Judul'].str.replace('Halaman all', '')

# Melakukan case folding pada teks
data['Judul'] = data['Judul'].str.lower()

# Menghapus stopwords pada teks
stop_words = set(stopwords.words('indonesian'))
data['Judul'] = data['Judul'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Melakukan tokenisasi pada teks tweet
data['Judul'] = data['Judul'].apply(lambda x: word_tokenize(x))

# Melakukan stemming pada teks
stemmer = StemmerFactory().create_stemmer()
data['Judul'] = data['Judul'].apply(lambda x: [stemmer.stem(word) for word in x])

# Menggabungkan token-token menjadi kalimat-kalimat kembali
data['Judul'] = data['Judul'].apply(' '.join)

# Inisialisasi objek TfidfVectorizer
vectorizer = TfidfVectorizer()

# Mengubah teks menjadi vektor TF-IDF
tfidf_matrix = vectorizer.fit_transform(data['Judul'])

# Pisahkan fitur (TF-IDF) dan label
X = tfidf_matrix
y = data['label']

# # Memisahkan fitur dan label
# X = data['Judul']
# y = data['label']

# # Membuat TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(X)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=1)

# Melatih model KNN
knn.fit(X_train, y_train)

# knn.fit(X, y)

#Melakukan prediksi pada data uji
y_pred = knn.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)

# Fungsi untuk mendeteksi berita
def detect_hoax(news_text):
    news_text = vectorizer.transform([news_text])
    prediction = knn.predict(news_text)
    return prediction[0]

# Tampilan web menggunakan Streamlit
news_input = st.text_area("Masukkan teks berita 👇🏼")
if st.button("Deteksi"):
    if news_input:
        result = detect_hoax(news_input)
        with st.spinner('Lagi Loading...'):
            time.sleep(5)
        st.write("Hasil Deteksi : ", result)
        st.write('Hasil Akurasi Model : {:.2f}% '.format(accuracy * 100))
    else:
        st.warning('Masukkan teks berita terlebih dahulu !')