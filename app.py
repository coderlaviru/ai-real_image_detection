<<<<<<< Updated upstream
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("real_vs_fake_cnn.h5")

IMG_HEIGHT = 128
IMG_WIDTH = 128

st.title("Real vs Fake Image Detector 🔍")
st.write("Upload any image and the model will predict if it is REAL or FAKE.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img = image.load_img(uploaded, target_size=(IMG_HEIGHT, IMG_WIDTH))
    st.image(img, caption='Uploaded Image', width=300)


    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred < 0.5:
        st.error("❌ FAKE IMAGE DETECTED!")
    else:
        st.success("✔️ REAL IMAGE DETECTED!")

    st.write("Prediction score:", float(pred))
=======
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("real_vs_fake_cnn.h5")

IMG_HEIGHT = 128
IMG_WIDTH = 128

st.title("Real vs Fake Image Detector 🔍")
st.write("Upload any image and the model will predict if it is REAL or FAKE.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img = image.load_img(uploaded, target_size=(IMG_HEIGHT, IMG_WIDTH))
    st.image(img, caption='Uploaded Image', width=300)


    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred < 0.5:
        st.error("❌ FAKE IMAGE DETECTED!")
    else:
        st.success("✔️ REAL IMAGE DETECTED!")

    st.write("Prediction score:", float(pred))
    
>>>>>>> Stashed changes
