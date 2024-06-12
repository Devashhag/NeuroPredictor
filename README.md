# Neuro Predictor by Devashish Agarwal
Neuro Predictor is a web application developed using Streamlit and Keras for classifying brain tumor types from MRI images. The application allows users to upload an MRI image and predicts the type of brain tumor present in the image.

## How it Works

The application uses convolutional neural networks (CNNs) implemented with Keras to classify brain tumor types. Three different CNN architectures, namely Inception, VGG16, and a custom CNN, were trained on a dataset of MRI images to perform the classification task. After evaluation, the VGG16 model was chosen as the best performing model for classification.

## Usage

Run the Streamlit app using the following command:

```bash
streamlit run app.py
