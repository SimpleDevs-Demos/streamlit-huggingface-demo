---
title: Streamlit + HuggingFace Demo
emoji: 🌍
colorFrom: pink
colorTo: yellow
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---

# Streamlit + HuggingFace Demo

This is an example repository that represents a viable way to structure a project and deploy it on either HuggingFace or Streamlit Community Pages.

## What is this project?

This is an example project that contains an example of Sentiment Analysis. The project uses **Streamlit** to render UI elements, while all models used here are run off of Transformers and Tokenizers from **HuggingFace**. The main interface is driven by `app.py`. If you want demos, visit this example **[running on Streamlit](https://simpledevs-huggingface-demo.streamlit.app/)** or **[running on HuggingFace](https://huggingface.co/spaces/rk2546/Streamlit-Huggingface-Demo)**.

Sentiment Analysis relies on pre-trained [models](https://huggingface.co/models) from HuggingFace's public [datasets](https://huggingface.co/datasets) - particularly 4 models:

- [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- [finiteautomata/beto-sentiment-analysis](https://huggingface.co/finiteautomata/beto-sentiment-analysis)
- [bhadresh-savani/distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)
- [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english)

If you want to run the code locally, then you need to set up a virtual environment. The general recommended way is to set up a Docker image and container, as this gives your application an environment that can be easily loaded into most cloud systems. A more prototype, local method is to create a python environment:

```bash
# Create your environment, which here we name ".venv"
python -m venv .venv

# Initialize your environment
.venv/Scripts/activate      # Windows
source .venv/bin/activate   # OS X, Linux

# Install pre-requisites
pip install -r requirements.txt

# Run the streamlit application locally
streamlit run app.py

# Deactivate the environment when you are done
deactivate
```

**Note**: If you decide to run this locally, note that you might want to have Torch and its related drivers installed onto your system beforehand.

## Code Explanation (For Streamlit and HuggingFace Newbies)

### Importing Streamlit and Transformers from HuggingFace

````python
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
````

### Loading a HuggingFace Model

```python
classifier = pipeline(
    model = AutoModelForSequenceClassification.from_pretrained(model_name), 
    tokenizer = AutoTokenizer.from_pretrained(model_name),
    padding=True, 
    truncation=True,
    task = "sentiment-analysis"
)
```

### User Interface

Streamlit gives you a lot of functions to render content as markdown. You may want to look at [the Streamlit Documentation](https://docs.streamlit.io/) prior to reading this part.

The interface is run on `app.py`. Here are some notable elements:

```python
st.title("Streamlit Huggingface Demo")
st.markdown("This is an example project for running Python-based projects using **Streamlit** and **HuggingFace**. You can also read our README (available via Github) to learn more about little details of this project.")
```

This prints out text onto the screen using Markdown syntax.

---

```python
model_option = st.selectbox(
    "What sentiment analysis model do you want to use? NOTE: Lag may occur when loading a new model!",
    _MODEL_NAMES,
    key="model_name",
    on_change=change_model
)
```

This is a dropdown box that lets you select between different models. Note that we are letting the user select items from `_MODEL_NAMES` and that the value is being stored in `st.session_state.model_name`. When the user changes this value in the dashboard, a function named `change_model()` is called.

---

```python
form = st.form(key='sentiment-analysis-form')
form.text_area(
    "Enter some text for sentiment analysis! If you just want to test it out without entering anything, just press the \"Submit\" button and the model will look at the placeholder.", 
    placeholder=_PLACEHOLDER,
    key="query"
)
submitted = form.form_submit_button('Submit', disabled=st.session_state.is_loading or st.session_state.classifier is None)
```

We create a semantic form and populate it with a text area. It contains a placeholder and saves the value inside of `st.session_state.query`. We also have a submit button that is **disabled** when a model change has occurred; we set that session state in `change_model()`. The state of the button is stored inside of `submitted` variables.

---

```python
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
```

The state of the submit button, `submittetd` is picked up in the next part. We do some pre-processing and replace the entry with the placeholder if the user hasn't entered anything. After, we call upon our classifier (which we stored in `st.session_state.classifier`) to give us a raw result. We parse that result using a custom function `parse_output()`, and then print the output to the screen. That function `output_func()` is returned by `parse_output()`, depending on the model used.