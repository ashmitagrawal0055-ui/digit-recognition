import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

model = load_model("mnist_model12new.h5")

def preprocess_image(image_data):
    img = Image.fromarray(image_data.astype(np.uint8))
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)

st.title("Predict a 3-digit number")
st.write("Draw one digit (0-9) on each canvas from left to right.")
st.write("Click Predict to get the 3-digit prediction and top-3 candidates.")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Digit 1")
    canvas1 = st_canvas(fill_color="black", stroke_width=25, stroke_color="black", background_color="white", height=280, width=280, drawing_mode="freedraw", key="canvas1")
with col2:
    st.markdown("### Digit 2")
    canvas2 = st_canvas(fill_color="black", stroke_width=25, stroke_color="black", background_color="white", height=280, width=280, drawing_mode="freedraw", key="canvas2")
with col3:
    st.markdown("### Digit 3")
    canvas3 = st_canvas(fill_color="black", stroke_width=25, stroke_color="black", background_color="white", height=280, width=280, drawing_mode="freedraw", key="canvas3")

if st.button("Predict"):
    canvases = [canvas1, canvas2, canvas3]
    images = []
    for c in canvases:
        if c is None or c.image_data is None:
            images = []
            break
        images.append(preprocess_image(c.image_data))
    if not images:
        st.error("Please draw on all three canvases before predicting.")
    else:
        prob = [] #per canvas probability
        pred = [] #single digit argmax prediction
        with st.spinner("Predicting..."):
            for img in images:
                p = model.predict(img)[0]
                prob.append(p)
                pred.append(str(np.argmax(p)))
            predicted_number = "".join(pred)
            st.success(f"Predicted 3-digit number: {predicted_number}")
            p0, p1, p2 = prob
            indices = np.arange(10)
            combos = []
            for i in indices:
                for j in indices:
                    for k in indices:
                        prob = p0[i] * p1[j] * p2[k]
                        combos.append((prob, f"{i}{j}{k}"))
            combos.sort(reverse=True, key=lambda x: x[0])
            st.markdown("### Top 3 Predictions")
            for prob, num in combos[:3]:
                st.write(f"**{num}** — {prob*100:.2f}%")
        st.success("Done")
