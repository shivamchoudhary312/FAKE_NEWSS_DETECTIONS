import streamlit as st
import pandas as pd
import pickle
import re
from urllib.parse import urlparse
from scipy.sparse import issparse
# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)
# =========================================================
# LOAD SAVED MODEL
# ========================================================
@st.cache_resource
def load_saved_model():
    # Load main model
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    # Try loading preprocessor/vectorizer
    preprocessor = None
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            preprocessor = pickle.load(f)
    except Exception:
        preprocessor = None
    return model, preprocessor
model, preprocessor = load_saved_model()
# =========================================================
# TEXT CLEANIN
# =========================================================
def clean_text(text):
    text = str(text).lower()
    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )
    # Keep only letters, numbers and spaces
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )
    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )
    return text.strip()
# =========================================================
# URL CLEANING
# =========================================================

def clean_url(url):

    url = str(url).lower().strip()

    url = re.sub(
        r"https?://",
        "",
        url
    )

    url = re.sub(
        r"www\.",
        "",
        url
    )

    url = re.sub(
        r"[^a-zA-Z0-9\s./_-]",
        " ",
        url
    )

    url = re.sub(
        r"\s+",
        " ",
        url
    )

    return url.strip()


# =========================================================
# DOMAIN CLEANING
# =========================================================

def clean_domain(url):

    url = str(url).lower().strip()

    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:

        domain = urlparse(url).netloc

        domain = domain.replace(
            "www.",
            ""
        )

        return domain

    except Exception:

        return url


# =========================================================
# CHECK IF OBJECT IS A PIPELINE
# =========================================================

def is_pipeline(obj):

    return hasattr(obj, "steps") and hasattr(
        obj,
        "predict"
    )


# =========================================================
# APP TITLE
# =========================================================

st.title("📰 Fake News Detector")

st.write(
    "Enter the news details below and the trained "
    "machine learning model will predict whether "
    "the news is **Real** or **Fake**."
)

st.divider()


# =========================================================
# INPUTS
# =========================================================

title = st.text_area(
    "📰 News Title",
    placeholder="Enter the news headline here...",
    height=120
)


url = st.text_input(
    "🔗 News URL",
    placeholder="https://example.com/news"
)


domain = st.text_input(
    "🌐 Source Domain",
    placeholder="example.com"
)


tweet_num = st.number_input(
    "🐦 Tweet Count",
    min_value=0,
    value=0,
    step=1
)


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔍 Detect News",
    use_container_width=True
):

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if not title.strip():

        st.warning(
            "⚠️ Please enter a news title."
        )

        st.stop()


    # -----------------------------------------------------
    # CLEAN INPUT
    # -----------------------------------------------------

    title_clean = clean_text(title)

    url_clean = clean_url(url)


    # If user gives domain
    if domain.strip():

        domain_clean = clean_domain(domain)

    # Otherwise extract domain from URL
    else:

        domain_clean = clean_domain(url)


    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "title_clean": [
            title_clean
        ],

        "url_clean": [
            url_clean
        ],

        "domain_clean": [
            domain_clean
        ],

        "tweet_num": [
            tweet_num
        ]

    })


    try:

        # =================================================
        # CASE 1
        # COMPLETE PIPELINE
        # =================================================

        if is_pipeline(model):

            prediction = model.predict(
                input_data
            )[0]


            # Probability
            if hasattr(
                model,
                "predict_proba"
            ):

                probability = model.predict_proba(
                    input_data
                )[0]

                confidence = (
                    max(probability) * 100
                )

            else:

                confidence = None


        # =================================================
        # CASE 2
        # MODEL + PREPROCESSOR
        # =================================================

        else:

            if preprocessor is None:

                st.error(
                    "Preprocessor file could not be loaded."
                )

                st.stop()


            # Transform original 4-column input
            X_input = preprocessor.transform(
                input_data
            )


            # Predict
            prediction = model.predict(
                X_input
            )[0]


            # Probability
            if hasattr(
                model,
                "predict_proba"
            ):

                probability = model.predict_proba(
                    X_input
                )[0]

                confidence = (
                    max(probability) * 100
                )

            else:

                confidence = None


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        # Convert prediction to string
        prediction_str = str(
            prediction
        ).lower().strip()


        # -------------------------------------------------
        # REAL
        # -------------------------------------------------

        if prediction_str in [
            "1",
            "1.0",
            "true",
            "real"
        ]:

            st.success(
                "## ✅ REAL NEWS"
            )


        # -------------------------------------------------
        # FAKE
        # -------------------------------------------------

        elif prediction_str in [
            "0",
            "0.0",
            "false",
            "fake"
        ]:

            st.error(
                "## ❌ FAKE NEWS"
            )


        # -------------------------------------------------
        # OTHER LABEL
        # -------------------------------------------------

        else:

            st.info(
                f"## Prediction: {prediction}"
            )


        # =================================================
        # CONFIDENCE
        # =================================================

        if confidence is not None:

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )


        # =================================================
        # INPUT SUMMARY
        # =================================================

        with st.expander(
            "🔎 View processed input"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Fake News Detection using NLP, TF-IDF, "
    "Machine Learning and Streamlit"
)