import os
import gradio as gr
import joblib

# Load the trained model once when the app starts
deployed_lr = joblib.load("my_first_ml_model.pkl")


def predict_rent(size_of_prop):
    # The model expects a 2D array: [[size]]
    prediction = deployed_lr.predict([[size_of_prop]])
    return f"Estimated Rent: ${prediction[0]:.2f}"


# Custom CSS
custom_css = """
.gradio-container {
    background-image: url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.glass-container {
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    color: #1f2937 !important;
}

.glass-container h1,
.glass-container h3,
.glass-container p,
.glass-container a,
.glass-container ul,
.glass-container li,
.glass-container strong {
    color: #1f2937 !important;
}

.glass-container a {
    color: #2563eb !important;
    text-decoration: none;
}

.glass-container a:hover {
    text-decoration: underline;
}
"""


with gr.Blocks(css=custom_css, title="Property Rent Predictor") as interface:
    with gr.Column(elem_classes="glass-container"):

        gr.Markdown(
            "<h1 style='text-align: center;'>🏙️ Property Rent Predictor</h1>"
        )

        gr.Markdown(
            "<p style='text-align: center;'>Enter the property size to get a rent estimate powered by Machine Learning.</p>"
        )

        gr.HTML("<hr style='border-color: #d1d5db;'>")

        with gr.Row():

            # Left Column - Predictor
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Estimation Tool")

                size_input = gr.Number(
                    label="Please Enter the Size of Your Property for Rent (sq ft)"
                )

                predict_btn = gr.Button(
                    "Predict Rent",
                    variant="primary"
                )

                rent_output = gr.Text(
                    label="Predicted Rent"
                )

                predict_btn.click(
                    fn=predict_rent,
                    inputs=size_input,
                    outputs=rent_output
                )

            # Right Column - About Developer
            with gr.Column(scale=1):

                gr.Markdown("### 👨‍💻 About")

                gr.Markdown("""
**Made By**

**Chintu (28240403)**
""")

                gr.Markdown("### 🛠️ Tools Used")

                gr.Markdown("""
- **Python**: Core programming language.
- **Gradio**: Interactive web interface framework.
- **Scikit-Learn**: Machine learning model training.
- **Joblib**: Model serialization and loading.
                """)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
