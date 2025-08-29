import os
import json
import tensorflow as tf
import gradio as gr
import pandas as pd
from PIL import Image

from inference import predict_ensemble_topk, get_clean_class_list


# --- Performance summaries (as measured during training/eval) ---
PERF = {
    "ResNet50": {
        "TRAIN": {"Accuracy": 0.9348, "Precision": 0.9400, "Recall": 0.9386, "F1 Score": 0.9389, "AUC": 0.9953},
        "VAL":   {"Accuracy": 0.7345, "Precision": 0.7646, "Recall": 0.7494, "F1 Score": 0.7512, "AUC": 0.9570},
        "TEST":  {"Accuracy": 0.7924, "Precision": 0.8122, "Recall": 0.7868, "F1 Score": 0.7873, "AUC": 0.9716},
    },
    "ResNet101": {
        "TRAIN": {"Accuracy": 0.9209, "Precision": 0.9280, "Recall": 0.9331, "F1 Score": 0.9290, "AUC": 0.9948},
        "VAL":   {"Accuracy": 0.7530, "Precision": 0.7698, "Recall": 0.7764, "F1 Score": 0.7673, "AUC": 0.9594},
        "TEST":  {"Accuracy": 0.7661, "Precision": 0.7717, "Recall": 0.7656, "F1 Score": 0.7635, "AUC": 0.9681},
    },
    "EfficientNetB0": {
        "TRAIN": {"Accuracy": 0.9385, "Precision": 0.9420, "Recall": 0.9390, "F1 Score": 0.9398, "AUC": 0.9954},
        "VAL":   {"Accuracy": 0.7530, "Precision": 0.7712, "Recall": 0.7549, "F1 Score": 0.7551, "AUC": 0.9592},
        "TEST":  {"Accuracy": 0.7787, "Precision": 0.7961, "Recall": 0.7787, "F1 Score": 0.7816, "AUC": 0.9697},
    },
    "VGG16": {
        "TRAIN": {"Accuracy": 0.9256, "Precision": 0.9314, "Recall": 0.9259, "F1 Score": 0.9275, "AUC": 0.9948},
        "VAL":   {"Accuracy": 0.7199, "Precision": 0.7357, "Recall": 0.7212, "F1 Score": 0.7234, "AUC": 0.9556},
        "TEST":  {"Accuracy": 0.7766, "Precision": 0.7900, "Recall": 0.7678, "F1 Score": 0.7660, "AUC": 0.9697},
    },
}


def perf_df_for(model_name: str) -> pd.DataFrame:
    rows = []
    for split in ["TRAIN","VAL","TEST"]:
        m = PERF[model_name][split]
        rows.append({"Split": split, **m})
    df = pd.DataFrame(rows, columns=["Split","Accuracy","Precision","Recall","F1 Score","AUC"])
    return df

def perf_summary_text() -> str:
    # highlight best on TEST for Accuracy and F1
    test_acc = {m: PERF[m]["TEST"]["Accuracy"] for m in PERF}
    test_f1  = {m: PERF[m]["TEST"]["F1 Score"] for m in PERF}
    best_acc_model = max(test_acc, key=test_acc.get)
    best_f1_model  = max(test_f1,  key=test_f1.get)
    return (
        f"**Best TEST Accuracy:** {best_acc_model} ({test_acc[best_acc_model]:.4f})  \n"
        f"**Best TEST F1 Score:** {best_f1_model} ({test_f1[best_f1_model]:.4f})"
    )



def perf_df_for(model_name: str) -> pd.DataFrame:
    rows = []
    for split in ["TRAIN","VAL","TEST"]:
        m = PERF[model_name][split]
        rows.append({"Split": split, **m})
    df = pd.DataFrame(rows, columns=["Split","Accuracy","Precision","Recall","F1 Score","AUC"])
    return df

def perf_summary_text() -> str:
    # highlight best on TEST for Accuracy and F1
    test_acc = {m: PERF[m]["TEST"]["Accuracy"] for m in PERF}
    test_f1  = {m: PERF[m]["TEST"]["F1 Score"] for m in PERF}
    best_acc_model = max(test_acc, key=test_acc.get)
    best_f1_model  = max(test_f1,  key=test_f1.get)
    return (
        f"**Best TEST Accuracy:** {best_acc_model} ({test_acc[best_acc_model]:.4f})  \n"
        f"**Best TEST F1 Score:** {best_f1_model} ({test_f1[best_f1_model]:.4f})"
    )



def run_inference(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    per_model, ensemble_top3, probs, latency_ms, classes_clean, per_model_probs = predict_ensemble_topk(
        image, topk=3, aggregate="mean"
    )

    # 1) Per-model table (as you already had)
    df_models = pd.DataFrame(per_model).sort_values("top1_prob", ascending=False).reset_index(drop=True)

    # 2) Winner banner
    best_row = df_models.iloc[0]
    winner_text = f"## 🏆 Most confident: **{best_row['top1_class']}**  \n*{best_row['model']}* with prob **{best_row['top1_prob']:.3f}**"

    # 3) VGG16 Top-3 table (instead of ensemble)
    # Find VGG16 key (robust to naming)
    vgg_key = None
    for k in per_model_probs.keys():
        if "vgg" in k.lower():
            vgg_key = k
            break

    if vgg_key is None:
        vgg_note = "### (VGG16 not loaded — showing ensemble Top-3 instead)"
        df_vgg = pd.DataFrame(ensemble_top3, columns=["Class", "Probability"])
    else:
        vgg_probs = per_model_probs[vgg_key]
        # build class→prob pairs, sorted desc, take top-3
        pairs = list(zip(classes_clean, vgg_probs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        # df_vgg = pd.DataFrame(pairs[:1], columns=["Class", "Probability"])
        # vgg_note = "### Based on our testing, **VGG16** had the best test metric.  \nVGG16 class probabilities for the uploaded image:"

    # 4) Classes list and latency
    classes_joined = " | ".join(classes_clean)
    classes_markdown = "The uploaded image is classified under these classes: "+ classes_joined
    latency_text = f"Latency: {latency_ms:.1f} ms"

    return winner_text, df_models, classes_markdown, latency_text




with gr.Blocks(title="♻️ Waste Classifier") as demo:
    gr.Markdown("# ♻️ Waste Classifier")

    with gr.Tabs():
        # --- Tab 1: Inference ---
        with gr.TabItem("Classify"):
            gr.Markdown("Upload an image **or** use your **camera**. The app shows each model’s top-1, the most confident model, and **VGG16**’s Top-3.")
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(label="Input Image", sources=["upload","webcam"], type="pil")
                    btn = gr.Button("Predict")
                with gr.Column():
                    winner_md = gr.Markdown()
                    models_df = gr.Dataframe(headers=["model","top1_class","top1_prob"], interactive=False)
                    # vgg_note_md = gr.Markdown()
                    # vgg_df = gr.Dataframe(headers=["Class","Probability"], interactive=False)
                    classes_md = gr.Markdown()
                    latency_md = gr.Markdown()
            btn.click(
                fn=run_inference,
                inputs=inp,
                outputs=[winner_md, models_df, classes_md, latency_md]
            )

        # --- Tab 2: Performance ---
        with gr.TabItem("Performance"):
            gr.Markdown("## 📊 Model Performance Summary")
            perf_summary = gr.Markdown(perf_summary_text())

            # Show all four in accordions
            with gr.Accordion("ResNet50", open=False):
                res50_df = gr.Dataframe(value=perf_df_for("ResNet50"), interactive=False)
            with gr.Accordion("ResNet101", open=False):
                res101_df = gr.Dataframe(value=perf_df_for("ResNet101"), interactive=False)
            with gr.Accordion("EfficientNetB0", open=False):
                eff_df = gr.Dataframe(value=perf_df_for("EfficientNetB0"), interactive=False)
            with gr.Accordion("VGG16", open=False):
                vgg_df_perf = gr.Dataframe(value=perf_df_for("VGG16"), interactive=False)

    gr.Markdown("— Powered by Keras Transfer Learning + Gradio —")




if __name__ == "__main__":
    demo.launch()