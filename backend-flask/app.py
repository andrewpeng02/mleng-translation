import os
import pickle
import uuid
from threading import Thread
import time

import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import numpy as np
import mlflow

from flask import Flask, request, jsonify
from prometheus_flask_exporter.multiprocess import GunicornPrometheusMetrics
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from shared.db_helper import (
    insert_into_model_execution,
    update_model_execution_user_label,
)

app = Flask(__name__, static_folder="dist", static_url_path="/")
metrics = GunicornPrometheusMetrics(app)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["2000 per day", "1000 per hour", "200 per minute"],
    storage_uri="memory://",
)

if "IN_DOCKER" in os.environ and os.environ["IN_DOCKER"]:
    mlflow.set_tracking_uri("http://mlflow:8000")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:8000")
client = mlflow.MlflowClient()


def load_best_model():
    model_uri = "models:/transformer-translation@champion"
    version = client.get_model_version_by_alias("transformer-translation", "champion")
    # run_id = client.get_model_version_by_alias(
    #     "transformer-translation", "champion"
    # ).run_id
    model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path="model")
    model = torch.jit.load(model_path + "data/model.pth")

    # data_drift_model_path = mlflow.artifacts.download_artifacts(
    #     run_id=run_id, artifact_path="detector", dst_path="detector"
    # )
    # data_drift_model = load_detector(data_drift_model_path)

    lang_model = spacy.load("en_core_web_sm")
    with open(model_path + "files/en/freq_list.pkl", "rb") as f:
        en_freq_list = pickle.load(f)
    with open(model_path + "files/fr/freq_list.pkl", "rb") as f:
        fr_freq_list = pickle.load(f)

    model.eval()
    return {
        "model": model,
        # "data_drift_model": data_drift_model,
        "lang_model": lang_model,
        "en_freq_list": en_freq_list,
        "fr_freq_list": fr_freq_list,
        "version": version.version,
        "last_updated": version.creation_timestamp,
    }


model_data = None
try:
    model_data = load_best_model()
except:
    print("Model not found, please register a model in mlflow")


def update_model_if_out_of_date():
    global model_data
    if (
        model_data["version"]
        != client.get_model_version_by_alias(
            "transformer-translation", "champion"
        ).version
    ):
        model_data = load_best_model()


def return_with_error(input, error, id):
    Thread(
        target=insert_into_model_execution,
        kwargs={"id": id, "input": input, "error": error},
    ).start()
    return {"error": error}, 400


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/feedback", methods=["POST"])
def feedback_endpoint():
    req = request.get_json()
    if "id" not in req:
        return {"error": "Must contain id field"}, 400
    if "feedback" not in req or len(req["feedback"]) == 0:
        return {"error": "Must contain feedback field"}, 400
    id = req["id"]
    feedback = req["feedback"]

    Thread(
        target=update_model_execution_user_label,
        kwargs={"id": id, "user_label": feedback},
    ).start()

    return {"status": "success"}, 201


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    st = time.time()
    global model_data

    if model_data is None:
        try:
            model_data = load_best_model()
        except Exception as e:
            return str(e), 500

    Thread(target=update_model_if_out_of_date).start()

    req = request.get_json()
    if "input" not in req:
        result = {
            "output": "",
            "version": model_data["version"],
            "last_updated": model_data["last_updated"],
        }
        return jsonify(result)
    orig_input = req["input"]

    input = orig_input.strip()
    if len(input) == 0:
        return return_with_error(orig_input, "Input must be a valid string")
    if len(input) > 3000:
        return return_with_error(orig_input, "Input must have 3000 characters or fewer")

    # Split input by sentence
    doc = model_data["lang_model"](input)
    sentences = [sent.text for sent in doc.sents]
    if len(sentences) == 0:
        return return_with_error(orig_input, "Input must be a valid string")
    elif len(sentences) > 30:
        return return_with_error(orig_input, "Input contains too many sentences")
    sentences[-1] = (
        sentences[-1] + "."
        if sentences[-1][-1] not in {".", "!", "?"}
        else sentences[-1]
    )

    # Generate output by sentence
    id = uuid.uuid4()
    output = []
    for sentence in sentences:
        pred_result = predict(sentence, model_data, id)
        if "error" in pred_result:
            return return_with_error(orig_input, pred_result["error"], id)

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
    Thread(
        target=insert_into_model_execution,
        kwargs={"id": id, "input": orig_input, "output": output_detokenized},
    ).start()

    result = {
        "id": id,
        "output": output_detokenized,
        "version": model_data["version"],
        "time": str(time.time() - st),
        "last_updated": model_data["last_updated"],
    }
    return jsonify(result)


max_seq_length = 96


def predict(sentence, model_data, id):
    model = model_data["model"]
    fr_freq_list = model_data["fr_freq_list"]
    sentence = tokenize(sentence, model_data["en_freq_list"], model_data["lang_model"])
    if len(sentence) > max_seq_length:
        return {"error": "Sentence is too long"}
    if percent_oov(sentence, model_data["en_freq_list"]["[OOV]"]) > 0.33:
        return {"error": "Sentence contains too many invalid characters or words"}

    # Detect data drift
    # data_drift_detection(sentence, id)

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


def data_drift_detection(sentence, id):
    data_drift_model = model_data["data_drift_model"]

    preds = data_drift_model.predict(np.array(sentence), return_test_stat=True)
    if preds["data"]["is_drift"]:
        app.logger.error(f"Data drift detected! ID {id}")


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
