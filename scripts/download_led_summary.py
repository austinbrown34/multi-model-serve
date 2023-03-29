from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

MODEL_VARIATIONS = {
    "base": "pszemraj/led-base-book-summary",
    "large": "pszemraj/led-large-book-summary",
}

def download(model_dir="models", variations=["base", "large"]):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    for variation in variations:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_VARIATIONS[variation])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_VARIATIONS[variation])
        model.save_pretrained(os.path.join(model_dir, f"led-summary-{variation}"))
        tokenizer.save_pretrained(os.path.join(model_dir, f"led-summary-{variation}"))


if __name__ == "__main__":
    download()