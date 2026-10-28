import gradio as gr
import numpy as np
import spaces

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
_model = None

def get_model():
    global _model
    if _model is None:
        import keras
        _model = keras.models.load_model("model_scratch.keras")
    return _model

@spaces.GPU(duration=60)
def predict(img):
    model = get_model()
    img = img.convert("RGB").resize((150, 150))
    x = np.array(img, dtype="float32")[None]
    p = model.predict(x, verbose=0)[0]
    return {c: float(v) for c, v in zip(CLASSES, p)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=6),
    api_name="predict",
    title="ScapeNet: Natural Scene Classifier",
)
demo.launch()
