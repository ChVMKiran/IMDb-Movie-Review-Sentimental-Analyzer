import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ------------------------------------------
# 🎨 Custom Styling with HTML + CSS
st.markdown('''
    <style>
        .stApp {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(90deg, rgba(39, 34, 163, 1) 0%, rgba(47, 131, 222, 1) 50%, rgba(50, 146, 184, 1) 100%);
            color: #000000;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #01257D;
            text-align: center;
            text-shadow: 0 0 6px #FFFFFF;
        }
        .subtitle {
            font-size: 18px;
            color: #CCCCCC;
            text-align: center;
            margin-bottom: 30px;
        }
        .stTextArea textarea {
            font-size: 16px;
            border-radius: 12px;
            padding: 10px;
            background: rgba(50, 146, 184, 0.95);
        }
        .stButton button {
            color: white;
            background-color: rgba(39, 34, 163, 0.65);
            border-radius: 10px;
            font-size: 18px;
        }
        .positiveMsg {
            background-color: rgba(0,120,0,0.6); 
            color:white; 
            font-size:18px; 
            font-weight: bold; 
            padding: 19px; 
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .negativeMsg {
            background-color: rgba(120,0,0,0.6); 
            color:white; 
            font-size:18px; 
            font-weight: bold; 
            padding: 19px; 
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .warningMsg {
            background-color: rgba(255,206,27, 0.75); 
            color:white; 
            font-size:18px; 
            font-weight: bold; 
            padding: 19px; 
            border-radius: 10px;
            margin-bottom: 10px;
        }
    </style>
''', unsafe_allow_html=True)

# ------------------------------------------
# 🧠 App Title & Description
st.markdown("<div class='title'>🎬 Movie Review Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>✨ Let the ML model classify your review as Positive or Negative!</div>", unsafe_allow_html=True)


# ------------------------------------------
# 🔍 Load Data & Train Model
@st.cache_data
def load_model():
    data = pd.read_csv("imdb_sorted.csv")
    x = data['review']
    y = data['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    cv = CountVectorizer(stop_words='english')
    x_train_vec = cv.fit_transform(x_train)
    x_test_vec = cv.transform(x_test)
    model = MultinomialNB()
    model.fit(x_train_vec, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test_vec))
    return cv, model, accuracy

cv, model, accuracy = load_model()

# ------------------------------------------
# 📝 User Input
st.subheader("📝 Write your movie review here:")
user_review = st.text_area("For example: This movie was absolutely amazing! A must watch.", height=200)

min_length = 20

# ------------------------------------------
# 🎯 Predict Button
if st.button("🔍 Analyze Review "):
    if user_review.strip() == "":
        st.markdown("<div class = 'warningMsg'>Please enter a review before submitting. 🥲</div>", unsafe_allow_html=True)
    elif len(user_review.strip()) < min_length:
        st.markdown(f"<div class = 'warningMsg'>Write a bit longer review of atleast {min_length} characters. 🙂</div>",unsafe_allow_html=True)
    else:
        transformed = cv.transform([user_review])
        prediction = model.predict(transformed)[0]

        if prediction.lower() == "positive":
            st.markdown("<div class = 'positiveMsg'>✅ Positive Review! 🎉 Good Choice, get ready for the show! 🤩</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class = 'negativeMsg'>❌ Negative Review! 😢 Better go for something better... </div>", unsafe_allow_html=True)

# ------------------------------------------
# 📊 Accuracy Display
with st.expander("📈 Show Model Accuracy"):
    st.markdown(f"<p style='color: black;'>Model is {accuracy * 100:.2f}% accurate based on the IMDb review dataset.</p>", unsafe_allow_html=True)

# ------------------------------------------
# 🔗 Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️‍🩹 by <b style='color:#BAFF39;  text-shadow: 0 0 3px #FFD43A;'>Ch V M Kiran</b></p.", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>🚀 Powered by Python, Streamlit, Scikit-learn & pandas</p>",
    unsafe_allow_html=True
)