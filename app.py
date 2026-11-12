
import streamlit as st
import joblib
import re

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="SMS Shield | AI Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1220, #111c30);
    color: #f1f5f9;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: #f1f5f9 !important;
}

.hero {
    text-align: center;
    padding: 25px 10px;
}

.hero-icon {
    font-size: 48px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #a7b4c8;
    font-size: 16px;
}

.info-card {
    background: #17243b;
    border: 1px solid #2b3c57;
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
}

.result-spam {
    background: #3b1825;
    border: 1px solid #fb7185;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.result-ham {
    background: #123329;
    border: 1px solid #4ade80;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.result-title {
    font-size: 25px;
    font-weight: 800;
}

.small-text {
    color: #a7b4c8;
    font-size: 13px;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}

textarea {
    border-radius: 12px !important;
}

.footer {
    text-align: center;
    color: #8291a8;
    font-size: 12px;
    padding-top: 35px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# LOAD MODEL AND VECTORIZER
# ---------------------------------

@st.cache_resource
def load_models():
    model = joblib.load("best_sms_spam_model.pkl")
    vectorizer = joblib.load("best_tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, tfidf = load_models()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.info("Make sure both .pkl files are in the project folder.")
    st.stop()


# ---------------------------------
# TEXT PREPROCESSING
# ---------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------

def predict_sms(message):
    cleaned_message = clean_text(message)
    message_vector = tfidf.transform([cleaned_message])

    prediction = model.predict(message_vector)[0]

    probabilities = model.predict_proba(message_vector)[0]
    classes = list(model.classes_)

    spam_probability = probabilities[classes.index("spam")] * 100
    ham_probability = probabilities[classes.index("ham")] * 100

    return prediction, spam_probability, ham_probability


# ---------------------------------
# HEADER
# ---------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-icon">🛡️</div>
    <div class="hero-title">SMS Shield</div>
    <div class="hero-subtitle">
        Intelligent SMS Spam Detection powered by Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ---------------------------------
# MESSAGE INPUT
# ---------------------------------

st.subheader("🔍 Analyze a Message")

st.write("Enter an SMS message below to check whether it is legitimate or spam.")

examples = {
    "Choose an example...": "",
    "Spam example": "Congratulations! You have won a free prize. Call now!",
    "Legitimate example": "Hey, are we still meeting tomorrow?"
}

selected_example = st.selectbox(
    "Try an example",
    list(examples.keys())
)

if st.button("Use selected example"):
    st.session_state.sms_input = examples[selected_example]

message = st.text_area(
    "Your SMS message",
    placeholder="Type or paste your SMS message here...",
    height=140,
    key="sms_input"
)

analyze = st.button(
    "🛡️ Analyze Message",
    type="primary",
    use_container_width=True
)


# ---------------------------------
# DISPLAY PREDICTION
# ---------------------------------

if analyze:

    if not message.strip():
        st.warning("Please enter an SMS message first.")

    else:
        prediction, spam_prob, ham_prob = predict_sms(message)

        st.markdown("---")
        st.subheader("Analysis Results")

        if prediction == "spam":

            st.markdown(f"""
            <div class="result-spam">
                <div class="result-title">🚨 SPAM DETECTED</div>
                <p>This message has been classified as spam.</p>
            </div>
            """, unsafe_allow_html=True)

            confidence = spam_prob

        else:

            st.markdown(f"""
            <div class="result-ham">
                <div class="result-title">✅ LEGITIMATE MESSAGE</div>
                <p>This message has been classified as ham.</p>
            </div>
            """, unsafe_allow_html=True)

            confidence = ham_prob

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam probability",
                f"{spam_prob:.2f}%"
            )

        with col2:
            st.metric(
                "Ham probability",
                f"{ham_prob:.2f}%"
            )

        st.write("**Estimated probability of predicted class**")
        st.progress(int(round(confidence)))

        st.caption(
            "These are model-estimated probabilities, "
            "not guarantees or necessarily calibrated confidence scores."
        )


# ---------------------------------
# MODEL INFORMATION
# ---------------------------------

st.markdown("---")
st.subheader("🤖 About This AI Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>97.97%</h3>
        <p class="small-text">Test accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>91.89%</h3>
        <p class="small-text">Spam F1-score</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <h4>Technology Stack</h4>
    <p>Python · Scikit-learn · TF-IDF · Logistic Regression · Streamlit</p>
    <h4>Dataset</h4>
    <p>UCI SMS Spam Collection</p>
    <h4>Classification</h4>
    <p>Spam and Ham (legitimate SMS messages)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    SMS Shield | Machine Learning Project<br>
    Developed by Abdoolsamad
</div>
""", unsafe_allow_html=True)
