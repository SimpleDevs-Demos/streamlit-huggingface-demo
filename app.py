# ======================================================
# IMPORTS
# ======================================================
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# ======================================================
# GLOBAL VARIABLES
# ======================================================
_PLACEHOLDER = "@AmericanAir just landed - 3hours Late Flight - and now we need to wait TWENTY MORE MINUTES for a gate! I'll pay you back for the delay with a coffee."
_MODEL_NAMES = [
    "cardiffnlp/twitter-roberta-base-sentiment",
    "finiteautomata/beto-sentiment-analysis",
    "bhadresh-savani/distilbert-base-uncased-emotion",
    "siebert/sentiment-roberta-large-english"
]


# ======================================================
# PREDICTION FUNCTIONS
# ======================================================

def parse_output(model_name, result):
    label = result[0]['label']
    score = result[0]['score']
    output_func = st.info
    if model_name == "cardiffnlp/twitter-roberta-base-sentiment":
        if label == "LABEL_0":
            label = "NEGATIVE"
            output_func = st.error
        elif label == "LABEL_2":
            label = "POSITIVE"
            output_func = st.success
        else:
            label = "NEUTRAL"
    elif model_name == "finiteautomata/beto-sentiment-analysis":
        if label == "NEG":
            label = "NEGATIVE"
            output_func = st.error
        elif label == "POS":
            label = "POSITIVE"
            output_func = st.success
        else:
            label = "NEUTRAL"
    elif model_name == "bhadresh-savani/distilbert-base-uncased-emotion":
        if label == "sadness":
            output_func = st.info
        elif label == "joy":
            output_func = st.success
        elif label == "love":
            output_func = st.success
        elif label == "anger":
            output_func = st.error
        elif label == "fear":
            output_func = st.info
        elif label == "surprise":
            output_func = st.error
        label = label.upper()
    elif model_name == "siebert/sentiment-roberta-large-english":
        if label == "NEGATIVE":
            output_func = st.error
        elif label == "POSITIVE":
            output_func = st.success
    return label, score, output_func

def change_model(new_model_name:str = None):
    if new_model_name is None: 
        new_model_name = st.session_state.model_name
    st.session_state.is_loading = True
    st.session_state.classifier = pipeline(
        model = AutoModelForSequenceClassification.from_pretrained(new_model_name), 
        tokenizer = AutoTokenizer.from_pretrained(new_model_name),
        padding=True, 
        truncation=True,
        task = "sentiment-analysis"
    )
    st.session_state.is_loading = False

# ======================================================
# SESSION STATE HANDLING
# ======================================================
if "is_loading" not in st.session_state:
    st.session_state.is_loading = True
if "classifier" not in st.session_state:
    change_model(_MODEL_NAMES[0])


# ======================================================
# DASHBOARD
# ======================================================

st.title("Streamlit Huggingface Demo")
st.markdown("This is an example project for running Python-based projects using **Streamlit** and **HuggingFace**. You can also read our README (available via Github) to learn more about little details of this project.")

st.header("Sentiment Analysis")

model_option = st.selectbox(
    "What sentiment analysis model do you want to use? NOTE: Lag may occur when loading a new model!",
    _MODEL_NAMES,
    key="model_name",
    on_change=change_model
)

form = st.form(key='sentiment-analysis-form')
form.text_area(
    "Enter some text for sentiment analysis! If you just want to test it out without entering anything, just press the \"Submit\" button and the model will look at the placeholder.", 
    placeholder=_PLACEHOLDER,
    key="query"
)
submitted = form.form_submit_button('Submit', disabled=st.session_state.is_loading or st.session_state.classifier is None)

if submitted:
     # Handle missing user input
    query = st.session_state.query.strip() if st.session_state.query is not None and len(st.session_state.query.strip())>0 else _PLACEHOLDER
    raw_result = st.session_state.classifier(
        query
    )
    label, score, output_func = parse_output(
        st.session_state.model_name, 
        raw_result
    )
    output_func("**{}**: {}".format(label, score))