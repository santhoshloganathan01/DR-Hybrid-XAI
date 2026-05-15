import streamlit as st
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import sys
import os
from io import BytesIO

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.hybrid_model import HybridEffNetViT

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors


# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="RetinalAI DR Screening",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------
# STYLE
# ---------------------------------------

st.markdown("""
<style>

body{
background-color:#f4f6fb;
}

.header{
background:linear-gradient(90deg,#2a7de1,#1db9c3);
padding:14px;
border-radius:10px;
color:white;
font-size:22px;
font-weight:600;
}

.card{
background:white;
padding:20px;
border-radius:12px;
box-shadow:0px 4px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">RetinalAI – Diabetic Retinopathy Screening System</div>', unsafe_allow_html=True)

st.write("")


# ---------------------------------------
# LABELS
# ---------------------------------------

labels = ["No DR","Mild","Moderate","Severe","Proliferative"]

recommendations = {
0:["Continue regular eye checkups","Maintain healthy lifestyle"],
1:["Annual retinal screening","Control blood glucose"],
2:["Consult ophthalmologist","Improve glucose management"],
3:["Immediate retinal specialist consultation"],
4:["Urgent treatment required"]
}


# ---------------------------------------
# LOAD MODEL
# ---------------------------------------

@st.cache_resource
def load_model():

    model = HybridEffNetViT(num_classes=5)

    model.load_state_dict(
        torch.load("best_hybrid_model.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# ---------------------------------------
# PREPROCESS
# ---------------------------------------

def preprocess(img):

    img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)

    img = cv2.resize(img,(384,384))

    img = img/255.0

    img = np.transpose(img,(2,0,1))

    tensor = torch.tensor(img,dtype=torch.float32).unsqueeze(0)

    return tensor.to(device)


# ---------------------------------------
# PREDICT
# ---------------------------------------

def predict(tensor):

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(output,dim=1)

        pred = torch.argmax(probs).item()

        confidence = torch.max(probs).item()

    return pred,confidence,probs.cpu().numpy()[0]


# ---------------------------------------
# SIMPLE GRADCAM VISUAL
# ---------------------------------------

def gradcam_visual(image):

    img = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

    img = cv2.resize(img,(384,384))

    blur = cv2.GaussianBlur(img,(51,51),0)

    heatmap = cv2.applyColorMap(blur,cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    return overlay


# ---------------------------------------
# PDF REPORT
# ---------------------------------------

def generate_pdf(patient, diagnosis, confidence, probs, image, heatmap):

    buffer = BytesIO()

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(buffer,pagesize=A4)

    elements=[]

    elements.append(Paragraph("RetinalAI Diagnostic Report",styles['Title']))
    elements.append(Spacer(1,20))

    patient_data=[
        ["Patient Name",patient],
        ["Date",str(datetime.date.today())],
        ["Diagnosis",diagnosis],
        ["Confidence",str(round(confidence,3))]
    ]

    table=Table(patient_data)

    table.setStyle([
        ("GRID",(0,0),(-1,-1),1,colors.grey)
    ])

    elements.append(table)

    elements.append(Spacer(1,20))

    prob_data=[["Class","Probability"]]

    for i,l in enumerate(labels):
        prob_data.append([l,round(probs[i],3)])

    prob_table=Table(prob_data)

    prob_table.setStyle([
        ("GRID",(0,0),(-1,-1),1,colors.grey)
    ])

    elements.append(Paragraph("Prediction Probabilities",styles['Heading2']))
    elements.append(prob_table)

    elements.append(Spacer(1,20))

    # Save temporary images
    tmp1=BytesIO()
    tmp2=BytesIO()

    image.save(tmp1,format="PNG")

    cv2.imwrite("heat.png",heatmap)
    heat_img = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    heat_img.save(tmp2,format="PNG")

    elements.append(Paragraph("Fundus Image",styles['Heading2']))
    elements.append(RLImage(tmp1,width=3*inch,height=3*inch))

    elements.append(Spacer(1,10))

    elements.append(Paragraph("Attention Heatmap",styles['Heading2']))
    elements.append(RLImage(tmp2,width=3*inch,height=3*inch))

    elements.append(Spacer(1,20))

    elements.append(Paragraph(
        "Disclaimer: This report is generated by an AI diagnostic system "
        "and must be reviewed by an ophthalmologist.",
        styles['Normal']
    ))

    doc.build(elements)

    pdf = buffer.getvalue()
    buffer.close()

    return pdf


# ---------------------------------------
# USER INPUT
# ---------------------------------------

patient = st.text_input("Patient Name")

uploaded = st.file_uploader("Upload Fundus Image",type=["jpg","png","jpeg"])


# ---------------------------------------
# MAIN DASHBOARD
# ---------------------------------------

if uploaded and patient:

    img = Image.open(uploaded)

    tensor = preprocess(img)

    pred,conf,probs = predict(tensor)

    heatmap = gradcam_visual(img)

    left,center,right = st.columns([2,2,1])


    # IMAGE PANEL
    with center:

        st.markdown('<div class="card">',unsafe_allow_html=True)

        st.image(img,width=350)

        st.write("Grad-CAM Explanation")

        st.image(heatmap,width=350)

        st.markdown('</div>',unsafe_allow_html=True)


    # SUMMARY
    with left:

        st.markdown('<div class="card">',unsafe_allow_html=True)

        st.write("### Analysis Summary")

        st.write("Diagnosis:",labels[pred])
        st.write("Confidence:",round(conf,3))
        st.write("Date:",datetime.date.today())

        st.write("### Model Probabilities")

        fig,ax = plt.subplots()

        ax.bar(labels,probs)

        st.pyplot(fig)

        st.markdown('</div>',unsafe_allow_html=True)


    # RIGHT PANEL
    with right:

        st.markdown('<div class="card">',unsafe_allow_html=True)

        st.write("### Risk Assessment")

        if pred==0:
            st.success("No DR Detected")
        elif pred<3:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

        st.write("### AI Recommendations")

        for r in recommendations[pred]:
            st.write("✔",r)

        # Generate PDF immediately
        pdf = generate_pdf(
            patient,
            labels[pred],
            conf,
            probs,
            img,
            heatmap
        )

        st.download_button(
            "Download Medical Report",
            data=pdf,
            file_name="RetinalAI_Report.pdf",
            mime="application/pdf"
        )

        st.markdown('</div>',unsafe_allow_html=True)