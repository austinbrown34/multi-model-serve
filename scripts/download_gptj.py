from transformers import GPTJForCausalLM, AutoTokenizer
import os


def download(model_dir: str = "models"):
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model.save_pretrained(os.path.join(model_dir, "gpt-j-6B"))
    tokenizer.save_pretrained(os.path.join(model_dir, "gpt-j-6B"))

if __name__ == "__main__":
    download()