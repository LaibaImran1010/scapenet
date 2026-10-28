import gradio as gr
import numpy as np
import spaces

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
FILES = {"scratch": "model_scratch.keras", "pretrained": "model_pretrained.keras"}
_models = {}

def get_model(name):
    if name not in _models:
        import keras
        _models[name] = keras.models.load_model(FILES[name])
    return _models[name]

@spaces.GPU(duration=60)
def predict(img, which):
    model = get_model(which)
    img = img.convert("RGB").resize((150, 150))
    x = np.array(img, dtype="float32")[None]
    p = model.predict(x, verbose=0)[0]
    return {c: float(v) for c, v in zip(CLASSES, p)}

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Radio(["scratch", "pretrained"], value="scratch", label="Model")],
    outputs=gr.Label(num_top_classes=6),
    api_name="predict",
    title="ScapeNet: Natural Scene Classifier",
)
demo.launch()