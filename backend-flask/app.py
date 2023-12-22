import os
import pickle
import re

import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import mlflow

from flask import Flask, request, jsonify
from prometheus_flask_exporter.multiprocess import GunicornPrometheusMetrics

app = Flask(__name__)
metrics = GunicornPrometheusMetrics(app)

if "IN_DOCKER" in os.environ and os.environ["IN_DOCKER"]:
    mlflow.set_tracking_uri("http://mlflow:8000")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:8000")
client = mlflow.MlflowClient()


def load_best_model():
    model_uri = "models:/transformer-translation@champion"
    version = client.get_model_version_by_alias(
        "transformer-translation", "champion"
    ).version
    model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path="model")
    model = torch.jit.load(model_path + "data/model.pth")

    lang_model = spacy.load("en_core_web_sm")
    with open(model_path + "files/en/freq_list.pkl", "rb") as f:
        en_freq_list = pickle.load(f)
    with open(model_path + "files/fr/freq_list.pkl", "rb") as f:
        fr_freq_list = pickle.load(f)

    model.eval()
    return {
        "model": model,
        "lang_model": lang_model,
        "en_freq_list": en_freq_list,
        "fr_freq_list": fr_freq_list,
        "version": version,
    }


model_data = None
try:
    model_data = load_best_model()
except:
    print("Model not found, please register a model in mlflow")


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    global model_data

    if (
        model_data is None
        or model_data["version"]
        != client.get_model_version_by_alias(
            "transformer-translation", "champion"
        ).version
    ):
        try:
            model_data = load_best_model()
        except Exception as e:
            return str(e), 500

    req = request.get_json()

    input = req["input"].strip()
    if input is None or len(input) == 0:
        return "Input must be a valid string", 400

    # Split input by sentence
    doc = model_data["lang_model"](input)
    sentences = [sent.text for sent in doc.sents]
    if len(sentences) == 0:
        return "Input must be a valid string", 400
    elif len(sentences) > 30:
        return "Input contains too many sentences", 400
    sentences[-1] = (
        sentences[-1] + "."
        if sentences[-1][-1] not in {".", "!", "?"}
        else sentences[-1]
    )

    # Generate output by sentence
    output = []
    for sentence in sentences:
        pred_result = predict(sentence, model_data)
        if "error" in pred_result:
            return pred_result["error"], 400

        pred = pred_result["pred"]
        if len(pred) > 0:
            pred[0] = pred[0][0].upper() + pred[0][1:]
            output.extend(pred)
    output = [word for word in output if word not in {"\u202f"}]
    output_detokenized = TreebankWordDetokenizer().detokenize(output)
    output_detokenized = (
        output_detokenized.replace("' ", "'")
        .replace(" -", "-")
        .replace("- ", "-")
        .replace(" ?", "?")
        .replace(" .", ".")
        .replace(" !", "!")
    )

    result = {"output": output_detokenized, "version": model_data["version"]}
    return jsonify(result)


max_seq_length = 96


def predict(sentence, model_data):
    model = model_data["model"]
    fr_freq_list = model_data["fr_freq_list"]
    sentence = tokenize(sentence, model_data["en_freq_list"], model_data["lang_model"])
    if len(sentence) > max_seq_length:
        return {"error": "Sentence is too long"}
    if percent_oov(sentence, model_data["en_freq_list"]["[OOV]"]) > 0.33:
        return {"error": "Sentence contains too many invalid characters or words"}

    # Generate the translated sentence, feeding the model's output into its input
    translated_sentence = [fr_freq_list["[SOS]"]]
    i = 0
    while int(translated_sentence[-1]) != fr_freq_list["[EOS]"] and i < max_seq_length:
        output = forward_model(model, sentence, translated_sentence, "cpu").to("cpu")
        values, indices = torch.topk(output, 5)
        translated_sentence.append(int(indices[-1][0]))

        i += 1

    # Return the translated sentence
    return {"pred": detokenize(translated_sentence, fr_freq_list)[1:-1]}


def percent_oov(tokenized_sentence, oov_token):
    count_oov = 0
    for token in tokenized_sentence:
        if token == oov_token:
            count_oov += 1
    return count_oov / len(tokenized_sentence)


def forward_model(model, src, tgt, device):
    src = torch.tensor(src).unsqueeze(0).long().to(device)
    tgt = torch.tensor(tgt).unsqueeze(0).to(device)
    tgt_mask = gen_nopeek_mask(tgt.shape[1]).to(device)

    src_padding = torch.zeros_like(src, dtype=torch.bool)
    tgt_padding = torch.zeros_like(tgt, dtype=torch.bool)

    output = model.forward(
        src,
        tgt,
        src_key_padding_mask=src_padding,
        tgt_key_padding_mask=tgt_padding,
        memory_key_padding_mask=src_padding,
        tgt_mask=tgt_mask,
    )

    return output.squeeze(0).to("cpu")


def tokenize(sentence, freq_list, lang_model):
    punctuation = ["(", ")", ":", '"', " "]

    sentence = sentence.lower()
    sentence = [
        tok.text
        for tok in lang_model.tokenizer(sentence)
        if tok.text not in punctuation
    ]
    return [
        freq_list[word] if word in freq_list else freq_list["[OOV]"]
        for word in sentence
    ]


def detokenize(sentence, freq_list):
    freq_list = {v: k for k, v in freq_list.items()}
    return [freq_list[token] for token in sentence]


def gen_nopeek_mask(length):
    mask = torch.transpose(torch.triu(torch.ones(length, length)) == 0, 0, 1)

    return mask


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=9696)
