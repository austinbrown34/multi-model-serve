import os
import logging
import json
import re
import time
from pathlib import Path
import nltk
from cleantext import clean
from summarize import load_model_and_tokenizer, summarize_via_tokenbatches
from utils import load_example_filenames, truncate_word_count

_here = Path(__file__).parent

nltk.download("stopwords")  # TODO=find where this requirement originates from

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def model_fn(model_dir):
    """
    Load the model for inference
    """

    model, tokenizer = load_model_and_tokenizer(model_dir)
    return {"model": model, "tokenizer": tokenizer}


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    model = model["model"]
    tokenizer = model["tokenizer"]
    input_text = input_data["input_text"]
    token_batch_length = int(input_data["params"].get("token_batch_length", 1024))
    model_size = input_data["params"].get("model_size", "large")
    max_input_length = input_data["params"].get("max_input_length", 1024)

    # Clean the text
    clean_text = clean(input_text, lower=False)
    settings = {
        "length_penalty": float(input_data["params"].get("length_penalty", 1.0)),
        "repetition_penalty": float(input_data["params"].get("repetition_penalty", 1.0)),
        "no_repeat_ngram_size": int(input_data["params"].get("no_repeat_ngram_size", 3)),
        "encoder_no_repeat_ngram_size": int(input_data["params"].get("encoder_no_repeat_ngram_size", 4)),
        "num_beams": int(input_data["params"].get("num_beams", 4)),
        "min_length": int(input_data["params"].get("min_length", 4)),
        "max_length": int(input_data["params"].get("max_length", token_batch_length // 4)),
        "early_stopping": bool(input_data["params"].get("early_stopping", True)),
        "do_sample": bool(input_data["params"].get("do_sample", True)),
    }
    st = time.perf_counter()
    history = {}
    max_input_length = 2048 if model_size == "base" else max_input_length
    processed = truncate_word_count(clean_text, max_input_length)

    if processed["was_truncated"]:
        tr_in = processed["truncated_text"]
        input_wc = re.split(r"\s+", input_text)
        msg = f"""
        <div style="background-color: #FFA500; color: white; padding: 20px;">
        <h3>Warning</h3>
        <p>Input text was truncated to {max_input_length} words. That's about {100*max_input_length/len(input_wc):.2f}% of the submission.</p>
        </div>
        """
        logging.warning(msg)
        history["WARNING"] = msg
    else:
        tr_in = input_text
        msg = None
    
    if len(input_text) < 50:
        msg = f"""
        <div style="background-color: #880808; color: white; padding: 20px;">
        <h3>Error</h3>
        <p>Input text is too short to summarize. Detected {len(input_text)} characters.
        Please load text by selecting an example from the dropdown menu or by pasting text into the text box.</p>
        </div>
        """
        logging.warning(msg)
        logging.warning("RETURNING EMPTY STRING")
        history["WARNING"] = msg

        return msg, "", []
    
    _summaries = summarize_via_tokenbatches(
        tr_in,
        model,
        tokenizer,
        batch_length=token_batch_length,
        **settings, 
    )
    sum_text = [f"Section {i}: " + s["summary"][0] for i, s in enumerate(_summaries)]
    sum_scores = [
        f" - Section {i}: {round(s['summary_score'],4)}"
        for i, s in enumerate(_summaries)
    ]

    sum_text_out = "\n".join(sum_text)
    history["Summary Scores"] = "<br><br>"
    scores_out = "\n".join(sum_scores)
    rt = round((time.perf_counter() - st) / 60, 2)
    print(f"Runtime: {rt} minutes")
    html = ""
    html += f"<p>Runtime: {rt} minutes on CPU</p>"
    if msg is not None:
        html += msg

    html += ""

    return {
        "html": html,
        "summary": sum_text_out,
        "scores": scores_out,
    }


    return model.__call__(input_data)

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    
    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """

    if response_content_type == "application/json":
        response_body = json.dumps(prediction)
    else:
        response_body = str(prediction)

    return response_body