## the prediction is correct and the code is working fine accordingly
import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
import base64
# Load the model
@st.cache_resource
def load_vgg16_model():
    return load_model("D:\\PROJECTS FINAL YEAR\\vgg16.h5")

model = load_vgg16_model()

# Title of the app

st.title("Neuro Predictor")

# Instructions
st.write("Upload an MRI image to classify the type of brain tumor.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Display the image
    st.image(img, channels="BGR")

    # Preprocess the image
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 224, 224, 3)

    # Predict
    
    try:
        a = model.predict(img_array)
        indices = a.argmax()

        # Display the result
        st.write("Prediction Result:")
        if indices == 0:
            st.write("Glioma Tumor")
            st.markdown("### Prevention for Glioma:")
            st.markdown("""
            - **Avoiding radiation exposure:** Limit exposure to ionizing radiation when possible.
            - **Maintaining a healthy lifestyle:** Balanced nutrition and regular exercise may support overall health.
            """)
        elif indices == 1:
            st.write("Meningioma Tumor")
            st.markdown("### Prevention for Meningioma:")
            st.markdown("""
            - **Limiting radiation exposure:** Minimize exposure to known carcinogens and ionizing radiation.
            - **Monitoring for symptoms:** Regular check-ups for early detection and treatment.
            """)
        elif indices == 2:
            st.write("No Tumor detection according to the model")
           
        elif indices == 3:
            st.write("Pituitary Tumor")
            st.markdown("### Prevention for Pituitary Tumor:")
            st.markdown("""
            - **Early treatment:** Addressing hormonal imbalances or genetic predispositions promptly.
            - **Regular medical monitoring:** Routine checks for hormonal abnormalities or neurological symptoms.
            """)
        else:
            st.write("Model not trained to identify this image")
    except Exception as e:
        st.write(f"An error occurred: {e}")
else:
    st.write("Please upload an image to get a prediction.")



def main():
    
    # Define content for each tumor type
    glioma_info = """
    **Glioma Tumor:**
    A glioma is a tumor that originates from the glial cells in the brain or spine. 
    These cells support and protect neurons and can become cancerous.
    """

    meningioma_info = """
    **Meningioma Tumor:**
    Meningiomas are tumors that arise from the meninges, the layers of tissue 
    that cover the brain and spinal cord. They are usually benign (non-cancerous) 
    and grow slowly.
    """

    pituitary_info = """
    **Pituitary Tumor:**
    Pituitary tumors develop in the pituitary gland, located at the base of the brain. 
    They can affect hormone production and cause various symptoms depending on their 
    size and hormone secretion.
    """

    # Sidebar with tumor information
    st.sidebar.title('Learn About:')
    selected_tumor = st.sidebar.radio(
        "",
        ("Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor")
    )

    if selected_tumor == "Glioma Tumor":
        st.sidebar.markdown(glioma_info)
    elif selected_tumor == "Meningioma Tumor":
        st.sidebar.markdown(meningioma_info)
    elif selected_tumor == "Pituitary Tumor":
        st.sidebar.markdown(pituitary_info)
        
    


if __name__ == "__main__":
    main()