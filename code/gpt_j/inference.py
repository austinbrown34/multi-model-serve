from transformers import GPTJForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import json

def model_fn(model_dir):
    """
    Load the model for inference
    """
    model = GPTJForCausalLM.from_pretrained(model_dir, revision="float16")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return {"model": model, "tokenizer": tokenizer}


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    model = model["model"]
    tokenizer = model["tokenizer"]
    prompt = input_data["prompt"]
    stop_sequence = input_data.get("stop_sequence", "</s>")
    pad_token_id = input_data.get("pad_token_id", 50256)
    params = input_data.get("params", {})
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    sens = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = (sens != pad_token_id).long()

    outputs = model.generate(
        sens,
        attention_mask=attention_mask,
        **params
    )

    if params.get("return_dict_in_generate", False):
        output_text += tokenizer.batch_decode(outputs.sequences)[0]
    else:
        output_text += tokenizer.batch_decode(outputs)[0]

    output_text = output_text.split(stop_sequence)[1]

    stop_index = output_text.indexOf(stop_sequence)
    if stop_index != -1:
        output_text = output_text.substring(0, stop_index)
    

    return {
        "generated_text": output_text,
    }


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