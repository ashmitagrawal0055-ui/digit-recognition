import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
smodel=load_model("mnist_model12new.h5")

st.title("Predict your digit!")
st.write("Draw a single digit number (0-9) in the box below and click 'Predict'.")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img=Image.fromarray(canvas_result.image_data.astype(np.uint8))
        img = img.convert("L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img_array = np.array(img).astype("float32")/255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        with st.spinner("Predicting image..."):
            prediction = smodel.predict(img_array)
            digit = np.argmax(prediction)
            st.success(f"Predicted digit: {digit}")
            prediction = prediction[0]
            top_3_indices = prediction.argsort()[-3:][::-1]
            st.markdown("Top 3 Predictions:")
            for i in top_3_indices:
                st.write(f"**Digit {i}** — {prediction[i]*100:.2f}% confidence")
        st.success("Prediction complete!")
    else:
        st.error("Please draw a digit before predicting.")
