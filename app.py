import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tamil Uyir Ezhuthu AI",
    page_icon="🔤",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0b1020;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00e5ff;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #b8c7d9;
    font-size: 18px;
    margin-bottom: 30px;
}

.result {
    text-align: center;
    padding: 25px;
    border-radius: 15px;
    background: #111a2e;
    border: 1px solid #00e5ff;
    margin-top: 20px;
}

.letter {
    font-size: 80px;
    font-weight: bold;
    color: #00e5ff;
}

.confidence {
    font-size: 22px;
    color: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">தமிழ் உயிரெழுத்து AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-Based Tamil Uyir Ezhuthu Recognition'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CLASS NAMES
# ============================================================

classes = [
    "அ",
    "ஆ",
    "இ",
    "ஈ",
    "உ",
    "ஊ",
    "எ",
    "ஏ",
    "ஐ",
    "ஒ",
    "ஓ"
]


# ============================================================
# CNN MODEL
# ============================================================

class UyirEzhuthuCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 32, 3, padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64, 3, padding=1
        )

        self.conv3 = nn.Conv2d(
            64, 128, 3, padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            128 * 8 * 8,
            256
        )

        self.fc2 = nn.Linear(
            256,
            num_classes
        )

        self.dropout = nn.Dropout(0.5)


    def forward(self, x):

        x = self.pool(
            F.relu(self.conv1(x))
        )

        x = self.pool(
            F.relu(self.conv2(x))
        )

        x = self.pool(
            F.relu(self.conv3(x))
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = UyirEzhuthuCNN(
        len(classes)
    )

    model.load_state_dict(
        torch.load(
            "uyir_ezhuthu_cnn.pth",
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (64, 64)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📷 Upload Tamil Letter")

uploaded_file = st.file_uploader(
    "Upload a single Uyir Ezhuthu image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("L")


    # Display uploaded image

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )


    # Transform

    input_tensor = transform(
        image
    )

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(
        device
    )


    # Prediction

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )


    predicted_index = (
        predicted_index.item()
    )

    confidence = (
        confidence.item() * 100
    )

    predicted_letter = classes[
        predicted_index
    ]


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="result">',
        unsafe_allow_html=True
    )

    st.write("### 🤖 AI Prediction")

    st.markdown(
        f'<div class="letter">{predicted_letter}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence">'
        f'Confidence: {confidence:.2f}%'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.write("### 📊 Class Probabilities")

    probability_values = (
        probabilities[0].cpu().numpy()
        * 100
    )

    for i, letter in enumerate(classes):

        st.write(
            f"{letter}: "
            f"{probability_values[i]:.2f}%"
        )

        st.progress(
            float(
                probability_values[i] / 100
            )
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Tamil Uyir Ezhuthu Recognition "
    "using a CNN trained from scratch"
)