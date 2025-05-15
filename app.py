from flask import Flask, render_template, request, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Load the model
model = tf.keras.models.load_model('model.h5')
class_names = ['Apple scab', 'Apple Black rot', 'Apple Cedar rust', 'Healthy Apple',
               'Cherry Powdery mildew', 'Healthy Cherry',
               'Corn Cercospora leaf spot Gray leaf spot', 'Corn(maize) Common rust',
               'Corn(maize) Northern Leaf Blight', 'Corn(maize) Healthy', 'Grape Black rot',
               'Grape Esca(Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 'Grape Healthy',
               'Peach Bacterial spot', 'Peach Healthy', 'Pepper bell Bacterial spot', 'Pepper bell Healthy',
               'Potato Early blight', 'Potato Late blight', 'Potato Healthy', 'Strawberry Leaf scorch',
               'Strawberry Healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
               'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two-spotted spider mite',
               'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato  mosaic virus',
               'Tomato Healthy']

# Disease descriptions
disease_data = {
    "Apple scab": {
        "description": "A fungal disease that causes dark, scabby lesions on apple leaves, fruit, and twigs.",
        "stage": "Early to Mid Growth Stage"
    },
    "Apple Black rot": {
        "description": "A fungal infection leading to black, circular lesions on apples and leaves.",
        "stage": "Mid to Late Growth Stage"
    },
    "Apple Cedar rust": {
        "description": "A fungal disease forming orange spore masses, affecting apple trees and junipers.",
        "stage": "Early Growth Stage"
    },
    "Healthy Apple": {
        "description": "No disease detected. The apple plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Cherry Powdery mildew": {
        "description": "A fungal infection that appears as a white powdery coating on cherry leaves and fruit.",
        "stage": "Flowering and Fruit Development Stage"
    },
    "Healthy Cherry": {
        "description": "No disease detected. The cherry plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Corn Cercospora leaf spot Gray leaf spot": {
        "description": "A fungal disease causing grayish leaf spots, leading to reduced photosynthesis.",
        "stage": "Vegetative Stage"
    },
    "Corn(maize) Common rust": {
        "description": "A fungal disease producing reddish-brown pustules on corn leaves.",
        "stage": "Vegetative to Reproductive Stage"
    },
    "Corn(maize) Northern Leaf Blight": {
        "description": "A fungal infection causing cigar-shaped lesions, leading to yield loss.",
        "stage": "Mid to Late Growth Stage"
    },
    "Corn(maize) Healthy": {
        "description": "No disease detected. The corn plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Grape Black rot": {
        "description": "A fungal disease causing dark spots on leaves and shriveling fruit.",
        "stage": "Fruit Development Stage"
    },
    "Grape Esca(Black Measles)": {
        "description": "A disease that affects grapevines, leading to black streaks and wilting.",
        "stage": "Mid to Late Growth Stage"
    },
    "Grape Leaf blight (Isariopsis Leaf Spot)": {
        "description": "A fungal disease causing irregular leaf spots and defoliation.",
        "stage": "Early to Mid Growth Stage"
    },
    "Grape Healthy": {
        "description": "No disease detected. The grape plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Peach Bacterial spot": {
        "description": "A bacterial infection causing sunken, dark lesions on peach fruits and leaves.",
        "stage": "Early to Mid Growth Stage"
    },
    "Peach Healthy": {
        "description": "No disease detected. The peach plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Pepper bell Bacterial spot": {
        "description": "A bacterial disease causing water-soaked lesions on leaves and fruits.",
        "stage": "Vegetative to Fruit Development Stage"
    },
    "Pepper bell Healthy": {
        "description": "No disease detected. The pepper plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Potato Early blight": {
        "description": "A fungal disease causing dark concentric rings on potato leaves.",
        "stage": "Early to Mid Growth Stage"
    },
    "Potato Late blight": {
        "description": "A severe fungal disease causing large, water-soaked lesions leading to crop loss.",
        "stage": "Late Growth Stage"
    },
    "Potato Healthy": {
        "description": "No disease detected. The potato plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Strawberry Leaf scorch": {
        "description": "A fungal disease causing brown, dried leaf edges, reducing fruit yield.",
        "stage": "Mid to Late Growth Stage"
    },
    "Strawberry Healthy": {
        "description": "No disease detected. The strawberry plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Tomato Bacterial spot": {
        "description": "A bacterial infection causing water-soaked spots on tomato leaves and fruit.",
        "stage": "Early to Mid Growth Stage"
    },
    "Tomato Early blight": {
        "description": "A fungal disease causing dark, target-like spots on lower tomato leaves.",
        "stage": "Mid to Late Growth Stage"
    },
    "Tomato Late blight": {
        "description": "A severe fungal disease causing large, dark lesions on leaves and stems.",
        "stage": "Late Growth Stage"
    },
    "Tomato Leaf Mold": {
        "description": "A fungal disease causing yellow spots on leaves, leading to mold growth.",
        "stage": "Mid to Late Growth Stage"
    },
    "Tomato Septoria leaf spot": {
        "description": "A fungal infection causing small, circular, brown spots on tomato leaves.",
        "stage": "Early to Mid Growth Stage"
    },
    "Tomato Spider mites Two-spotted spider mite": {
        "description": "An infestation of tiny spider mites causing leaf bronzing and defoliation.",
        "stage": "All Growth Stages"
    },
    "Tomato Target Spot": {
        "description": "A fungal disease causing circular leaf lesions with a dark center.",
        "stage": "Early to Mid Growth Stage"
    },
    "Tomato Tomato Yellow Leaf Curl Virus": {
        "description": "A viral disease causing yellow, curled leaves and stunted growth.",
        "stage": "Early Growth Stage"
    },
    "Tomato mosaic virus": {
        "description": "A viral infection leading to mottled, yellowed tomato leaves.",
        "stage": "Seedling to Vegetative Stage"
    },
    "Tomato Healthy": {
        "description": "No disease detected. The tomato plant is in a healthy condition.",
        "stage": "All Growth Stages"
    }
}


IMAGE_SIZE = 255

# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    lang = request.args.get('lang', 'en')
    translations = LANGUAGES.get(lang, LANGUAGES['en'])
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part', translations=translations, lang=lang)

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file', translations=translations, lang=lang)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # Read and preprocess the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            # Get prediction
            predicted_class, confidence = predict(img)

            # Retrieve disease details (description + stage)
            disease_info = disease_data.get(predicted_class, {"description": "No description available", "stage": "Unknown"})
            description = disease_info["description"]
            stage = disease_info["stage"]

            # Store prediction in session for report download
            session['latest_prediction'] = {
                'report_id': filename,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'disease': predicted_class,
                'confidence': confidence,
                'growth_stage': stage,
                'description': description,
                'etiology': f"Etiology information for {predicted_class}",
                'treatment_steps': "See treatment recommendations in the app.",
                'cultural_practices': "See cultural practices in the app.",
                'severity': 'N/A',
                'technician': 'AI System',
                'image_path': filepath
            }

            return render_template(
                'index.html',
                image_path=filepath,
                predicted_label=predicted_class,
                confidence=confidence,
                description=description,
                stage=stage,
                translations=translations,
                lang=lang
            )

    return render_template('index.html', message='Upload an image', translations=translations, lang=lang)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/research-paper')
def research_paper():
    return send_file('static/research_paper.pdf', as_attachment=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Utility function to generate PDF report
def generate_pdf_report(report_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=30)
    styles = getSampleStyleSheet()
    elements = []

    # Header
    header_style = ParagraphStyle('Header', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=18, spaceAfter=12)
    elements.append(Paragraph('Plant Pathology Disease Detection Report', header_style))
    elements.append(Spacer(1, 12))

    # Report Info Table (vertical alignment, no overlap, bold field names)
    info_data = [
        [Paragraph('<b>Report ID:</b>', styles['Normal']), report_data['report_id']],
        [Paragraph('<b>Date:</b>', styles['Normal']), report_data.get('date', '')],
        [Paragraph('<b>Status:</b>', styles['Normal']), 'Completed'],
        [Paragraph('<b>Laboratory:</b>', styles['Normal']), 'Plant Pathology Lab, AI Division'],
        [Paragraph('<b>Technician:</b>', styles['Normal']), report_data.get('technician', 'AI System')],
    ]
    info_table = Table(info_data, colWidths=[110, 350])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (0,0), (0,-1), colors.darkblue),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 10))

    # Executive Summary
    elements.append(Paragraph('<b>Executive Summary</b>', styles['Heading3']))
    summary = f"""
    <b>Disease Detected:</b> {report_data['disease']}<br/>
    <b>Confidence Level:</b> {report_data['confidence']}%<br/>
    <b>Growth Stage:</b> {report_data['growth_stage']}<br/>
    """
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1, 8))

    # Sample Information
    elements.append(Paragraph('<b>Sample Information</b>', styles['Heading3']))
    sample_info = f"""
    <b>Sample Type:</b> Plant Leaf<br/>
    <b>Analysis Method:</b> AI-based Image Analysis<br/>
    <b>Detection Method:</b> Deep Learning Model<br/>
    <b>Confidence Level:</b> {report_data['confidence']}%<br/>
    <b>Date of Analysis:</b> {report_data.get('date', '')}<br/>
    """
    elements.append(Paragraph(sample_info, styles['Normal']))
    elements.append(Spacer(1, 8))

    # Detailed Analysis
    elements.append(Paragraph('<b>Detailed Analysis</b>', styles['Heading3']))
    details = f"""
    <b>Disease Name:</b> {report_data['disease']}<br/>
    <b>Growth Stage:</b> {report_data['growth_stage']}<br/>
    <b>Description:</b> {report_data['description']}<br/>
    <b>Analysis Basis:</b> AI model image analysis.<br/>
    """
    elements.append(Paragraph(details, styles['Normal']))
    elements.append(Spacer(1, 8))

    # Sample Image Analysis
    elements.append(Paragraph('<b>Sample Image Analysis</b>', styles['Heading3']))
    if report_data.get('image_path'):
        try:
            elements.append(RLImage(report_data['image_path'], width=2.5*inch, height=2.5*inch))
        except Exception:
            elements.append(Paragraph('Image could not be loaded.', styles['Normal']))
    else:
        elements.append(Paragraph('No image available.', styles['Normal']))
    elements.append(Spacer(1, 8))

    # Confidence Pie Chart (left-aligned)
    elements.append(Spacer(1, 8))
    elements.append(Paragraph('<b>Detection Confidence</b>', styles['Heading3']))
    elements.append(Spacer(1, 4))
    drawing_width = 200
    pie_size = 80
    drawing = Drawing(drawing_width, 100)
    pie = Pie()
    pie.x = 0
    pie.y = 10
    pie.width = pie_size
    pie.height = pie_size
    pie.data = [float(report_data['confidence']), 100-float(report_data['confidence'])]
    pie.labels = [f"Detected ({report_data['confidence']}%)", f"Other ({100-float(report_data['confidence']):.1f}%)"]
    pie.slices.strokeWidth = 0.5
    pie.slices[0].fillColor = colors.green
    pie.slices[1].fillColor = colors.lightgrey
    drawing.add(pie)
    elements.append(drawing)
    elements.append(Spacer(1, 16))
    elements.append(Paragraph(f'<b>Detection Confidence:</b> {report_data["confidence"]}%', styles['Normal']))
    elements.append(Spacer(1, 8))

    # Treatment Recommendations
    elements.append(Paragraph('<b>Treatment Recommendations</b>', styles['Heading3']))
    treat = f"""
    <b>Etiology:</b> {report_data['etiology']}<br/>
    <b>Treatment Protocol:</b> {report_data['treatment_steps']}<br/>
    <b>Cultural Practices:</b> {report_data['cultural_practices']}<br/>
    """
    elements.append(Paragraph(treat, styles['Normal']))
    elements.append(Spacer(1, 8))

    # Analysis Results
    elements.append(Paragraph('<b>Analysis Results</b>', styles['Heading3']))
    results = f"""
    <b>Disease Severity:</b> {report_data.get('severity', 'N/A')}<br/>
    <b>Probability of Disease Presence:</b> {report_data['confidence']}%<br/>
    """
    elements.append(Paragraph(results, styles['Normal']))
    elements.append(Spacer(1, 8))

    # Treatment Timeline
    elements.append(Paragraph('<b>Treatment Timeline</b>', styles['Heading3']))
    timeline_data = [
        ['Day/Week', 'Action/Step'],
        ['Day 1', 'Begin treatment protocol'],
        ['Day 3', 'Monitor plant response'],
        ['Week 1', 'Apply follow-up treatment'],
        ['Week 2', 'Reassess and adjust as needed']
    ]
    timeline_table = Table(timeline_data, colWidths=[70, 300])
    timeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(timeline_table)
    elements.append(Spacer(1, 8))

    # Important Notes
    elements.append(Paragraph('<b>Important Notes</b>', styles['Heading3']))
    notes = """
    <ul>
    <li>Always follow local agricultural guidelines.</li>
    <li>Use protective equipment when applying treatments.</li>
    <li>Dispose of infected plant material safely.</li>
    </ul>
    """
    elements.append(Paragraph(notes, styles['Normal']))
    elements.append(Spacer(1, 8))

    # Conclusion
    elements.append(Paragraph('<b>Conclusion</b>', styles['Heading3']))
    conclusion = f"The AI-powered analysis detected {report_data['disease']} with {report_data['confidence']}% confidence at the {report_data['growth_stage']}. Prompt action is advised."
    elements.append(Paragraph(conclusion, styles['Normal']))
    elements.append(Spacer(1, 8))

    # Disclaimer
    elements.append(Paragraph('<b>Disclaimer</b>', styles['Heading3']))
    disclaimer = "This report is generated by an AI-powered system and is intended for informational purposes only. For severe or persistent issues, consult a certified plant pathologist or agricultural extension officer. The laboratory and AI provider are not liable for any direct or indirect consequences arising from the use of this report."
    elements.append(Paragraph(disclaimer, styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

@app.route('/report/<report_id>')
def report(report_id):
    # Use the latest prediction from the session
    report_data = session.get('latest_prediction')
    if not report_data:
        # If no prediction in session, redirect to home
        return redirect(url_for('home'))
    # Update report_id in case user requests a different one
    report_data['report_id'] = report_id
    pdf_buffer = generate_pdf_report(report_data)
    return send_file(pdf_buffer, as_attachment=True, download_name=f"PlantPathologyReport_{report_id}.pdf", mimetype='application/pdf')

LANGUAGES = {
    'en': {
        'title': 'Plant Pathology',
        'subtitle': 'AI-powered Crop Disease Detection',
        'upload_label': 'Upload a plant leaf image',
        'predict_btn': 'Predict',
        'disease': 'Disease',
        'desc': 'Description',
        'stage': 'Occurs During',
        'show_accuracy': 'Show Accuracy',
        'accuracy': 'Accuracy',
        'download_report': 'Download Report (PDF)',
        'share': 'Share',
        'recent_predictions': 'Recent Predictions',
        'no_predictions': 'No predictions yet. Upload an image to get started.',
        'current_session': 'Current Session',
        'confidence_text': 'confidence',
        'tab_prevention': 'Prevention',
        'tab_treatment': 'Treatment',
        'tab_symptoms': 'Symptoms',
        'tab_resources': 'Resources',
        'prev_methods': [
            'Maintain proper plant spacing for good air circulation',
            'Water plants at the base to avoid wetting leaves',
            'Use disease-resistant plant varieties when available',
            'Practice crop rotation to prevent disease buildup',
            'Keep garden tools clean and disinfected',
        ],
        'prev_cultural_title': 'Cultural Practices',
        'prev_cultural_intro': 'Implement these cultural practices to prevent disease spread:',
        'prev_cultural_list': [
            'Remove and destroy infected plant parts',
            'Maintain proper soil pH and fertility',
            'Use mulch to prevent soil splash',
            'Monitor plants regularly for early signs',
        ],
        'treat_options_title': 'Treatment Options',
        'treat_chemical': 'Use appropriate fungicides or bactericides as recommended by local agricultural authorities',
        'treat_organic': 'Neem oil, copper fungicides, or biological control agents',
        'treat_physical': 'Prune affected areas and maintain proper plant hygiene',
        'symptoms_common_title': 'Common Symptoms',
        'symptoms_common_list': [
            'Leaf spots or lesions',
            'Yellowing or browning of leaves',
            'Wilting or drooping',
            'Stunted growth',
            'Abnormal leaf patterns',
        ],
        'symptoms_help_title': 'When to Seek Help',
        'symptoms_help_intro': 'Contact a plant pathologist or agricultural expert if:',
        'symptoms_help_list': [
            'Symptoms persist after treatment',
            'Disease spreads rapidly',
            'Multiple plants are affected',
            'Unusual symptoms appear',
        ],
        'resources_additional_title': 'Additional Resources',
        'resources_research': 'Access scientific literature on disease management',
        'resources_expert': 'Connect with agricultural experts',
        'resources_community': 'Join farming communities for advice',
    },
    'hi': {
        'title': 'पौध रोग विज्ञान',
        'subtitle': 'एआई-संचालित फसल रोग पहचान',
        'upload_label': 'पौधे की पत्ती की छवि अपलोड करें',
        'predict_btn': 'पूर्वानुमान',
        'disease': 'रोग',
        'desc': 'विवरण',
        'stage': 'किस चरण में होता है',
        'show_accuracy': 'सटीकता दिखाएँ',
        'accuracy': 'सटीकता',
        'download_report': 'रिपोर्ट डाउनलोड करें (PDF)',
        'share': 'साझा करें',
        'recent_predictions': 'हाल की भविष्यवाणियाँ',
        'no_predictions': 'अभी तक कोई भविष्यवाणी नहीं। आरंभ करने के लिए एक छवि अपलोड करें।',
        'current_session': 'वर्तमान सत्र',
        'confidence_text': 'सटीकता',
        'tab_prevention': 'रोकथाम',
        'tab_treatment': 'उपचार',
        'tab_symptoms': 'लक्षण',
        'tab_resources': 'संसाधन',
        'prev_methods': [
            'अच्छे वायु संचार के लिए पौधों के बीच उचित दूरी बनाए रखें',
            'पत्तियों को गीला करने से बचने के लिए पौधों की जड़ों में पानी दें',
            'जहाँ उपलब्ध हो, रोग-प्रतिरोधी पौध किस्मों का उपयोग करें',
            'रोग संचयन को रोकने के लिए फसल चक्र अपनाएँ',
            'बागवानी उपकरणों को साफ और कीटाणुरहित रखें',
        ],
        'prev_cultural_title': 'सांस्कृतिक अभ्यास',
        'prev_cultural_intro': 'रोग फैलाव को रोकने के लिए ये सांस्कृतिक अभ्यास अपनाएँ:',
        'prev_cultural_list': [
            'संक्रमित पौध भागों को हटाएँ और नष्ट करें',
            'मिट्टी का pH और उर्वरता उचित रखें',
            'मिट्टी के छींटों से बचने के लिए मल्च का उपयोग करें',
            'प्रारंभिक लक्षणों के लिए पौधों की नियमित निगरानी करें',
        ],
        'treat_options_title': 'उपचार विकल्प',
        'treat_chemical': 'स्थानीय कृषि अधिकारियों की सिफारिश के अनुसार उपयुक्त फफूंदनाशी या जीवाणुनाशी का उपयोग करें',
        'treat_organic': 'नीम का तेल, तांबे के फफूंदनाशी, या जैविक नियंत्रण एजेंट',
        'treat_physical': 'प्रभावित हिस्सों की छंटाई करें और पौध स्वच्छता बनाए रखें',
        'symptoms_common_title': 'सामान्य लक्षण',
        'symptoms_common_list': [
            'पत्तियों पर धब्बे या घाव',
            'पत्तियों का पीला या भूरा होना',
            'मुरझाना या झुकना',
            'विकास में रुकावट',
            'पत्तियों के असामान्य पैटर्न',
        ],
        'symptoms_help_title': 'कब सहायता लें',
        'symptoms_help_intro': 'इन स्थितियों में पौध रोग विशेषज्ञ या कृषि विशेषज्ञ से संपर्क करें:',
        'symptoms_help_list': [
            'उपचार के बाद भी लक्षण बने रहें',
            'रोग तेजी से फैल रहा हो',
            'कई पौधे प्रभावित हों',
            'असामान्य लक्षण दिखें',
        ],
        'resources_additional_title': 'अतिरिक्त संसाधन',
        'resources_research': 'रोग प्रबंधन पर वैज्ञानिक साहित्य प्राप्त करें',
        'resources_expert': 'कृषि विशेषज्ञों से संपर्क करें',
        'resources_community': 'सलाह के लिए किसान समुदायों से जुड़ें',
    },
    'pa': {
        'title': 'ਪੌਦੇ ਦੀ ਰੋਗ ਵਿਗਿਆਨ',
        'subtitle': 'ਏਆਈ-ਚਲਿਤ ਫਸਲ ਰੋਗ ਪਛਾਣ',
        'upload_label': 'ਪੌਦੇ ਦੀ ਪੱਤੀ ਦੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ',
        'predict_btn': 'ਭਵਿੱਖਬਾਣੀ',
        'disease': 'ਰੋਗ',
        'desc': 'ਵੇਰਵਾ',
        'stage': 'ਕਿਸ ਪੜਾਅ ਦੌਰਾਨ',
        'show_accuracy': 'ਸਹੀਤਾ ਵੇਖਾਓ',
        'accuracy': 'ਸਹੀਤਾ',
        'download_report': 'ਰਿਪੋਰਟ ਡਾਊਨਲੋਡ ਕਰੋ (PDF)',
        'share': 'ਸਾਂਝਾ ਕਰੋ',
        'recent_predictions': 'ਤਾਜ਼ਾ ਭਵਿੱਖਬਾਣੀਆਂ',
        'no_predictions': 'ਹਾਲੇ ਕੋਈ ਭਵਿੱਖਬਾਣੀ ਨਹੀਂ। ਸ਼ੁਰੂ ਕਰਨ ਲਈ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ।',
        'current_session': 'ਮੌਜੂਦਾ ਸੈਸ਼ਨ',
        'confidence_text': 'ਸਹੀਤਾ',
        'tab_prevention': 'ਰੋਕਥਾਮ',
        'tab_treatment': 'ਉਪਚਾਰ',
        'tab_symptoms': 'ਲੱਛਣ',
        'tab_resources': 'ਸਰੋਤ',
        'prev_methods': [
            'ਚੰਗੀ ਹਵਾ ਲਈ ਪੌਦਿਆਂ ਵਿਚਕਾਰ ਢੁਕਵੀਂ ਦੂਰੀ ਰੱਖੋ',
            'ਪੱਤਿਆਂ ਨੂੰ ਭਿੱਜਣ ਤੋਂ ਬਚਾਉਣ ਲਈ ਜੜਾਂ ਉੱਤੇ ਪਾਣੀ ਦਿਓ',
            'ਜਿੱਥੇ ਉਪਲਬਧ ਹੋਵੇ, ਰੋਗ-ਰੋਧੀ ਕਿਸਮਾਂ ਵਰਤੋ',
            'ਰੋਗ ਇਕੱਠਾ ਹੋਣ ਤੋਂ ਬਚਣ ਲਈ ਫਸਲ ਚੱਕਰ ਅਪਣਾਓ',
            'ਬਾਗਬਾਨੀ ਦੇ ਸਾਜ਼ੋ-ਸਾਮਾਨ ਨੂੰ ਸਾਫ਼ ਅਤੇ ਜਰਾਸੀਮ ਰਹਿਤ ਰੱਖੋ',
        ],
        'prev_cultural_title': 'ਸੱਭਿਆਚਾਰਕ ਅਭਿਆਸ',
        'prev_cultural_intro': 'ਰੋਗ ਦੇ ਫੈਲਾਅ ਨੂੰ ਰੋਕਣ ਲਈ ਇਹ ਅਭਿਆਸ ਅਪਣਾਓ:',
        'prev_cultural_list': [
            'ਸੰਕ੍ਰਮਿਤ ਹਿੱਸਿਆਂ ਨੂੰ ਹਟਾਓ ਅਤੇ ਨਸ਼ਟ ਕਰੋ',
            'ਮਿੱਟੀ ਦਾ pH ਅਤੇ ਉਪਜਾਊਪਨ ਠੀਕ ਰੱਖੋ',
            'ਮਿੱਟੀ ਦੇ ਛਿਟੇ ਤੋਂ ਬਚਣ ਲਈ ਮਲਚ ਵਰਤੋ',
            'ਪੌਦਿਆਂ ਦੀ ਨਿਯਮਤ ਜਾਂਚ ਕਰੋ',
        ],
        'treat_options_title': 'ਉਪਚਾਰ ਵਿਕਲਪ',
        'treat_chemical': 'ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਅਧਿਕਾਰੀਆਂ ਦੀ ਸਿਫਾਰਸ਼ ਅਨੁਸਾਰ ਉਚਿਤ ਫਫੂੰਦਨਾਸ਼ਕ ਜਾਂ ਬੈਕਟੀਰੀਆਨਾਸ਼ਕ ਵਰਤੋ',
        'treat_organic': 'ਨੀਮ ਦਾ ਤੇਲ, ਤਾਂਬੇ ਦੇ ਫਫੂੰਦਨਾਸ਼ਕ ਜਾਂ ਜੈਵਿਕ ਏਜੰਟ',
        'treat_physical': 'ਪ੍ਰਭਾਵਿਤ ਹਿੱਸਿਆਂ ਦੀ ਛੰਟਾਈ ਕਰੋ ਅਤੇ ਪੌਦੇ ਦੀ ਸਫਾਈ ਰੱਖੋ',
        'symptoms_common_title': 'ਆਮ ਲੱਛਣ',
        'symptoms_common_list': [
            'ਪੱਤਿਆਂ ਉੱਤੇ ਧੱਬੇ ਜਾਂ ਘਾਅ',
            'ਪੱਤਿਆਂ ਦਾ ਪੀਲਾ ਜਾਂ ਭੂਰਾ ਹੋਣਾ',
            'ਮੁਰਝਾਉਣਾ ਜਾਂ ਝੁਕਣਾ',
            'ਵਾਧੂ ਵਿੱਚ ਰੁਕਾਵਟ',
            'ਪੱਤਿਆਂ ਦੇ ਅਜਿਹੇ ਪੈਟਰਨ',
        ],
        'symptoms_help_title': 'ਮਦਦ ਕਦੋਂ ਲੈਣੀ',
        'symptoms_help_intro': 'ਇਨ੍ਹਾਂ ਹਾਲਾਤਾਂ ਵਿੱਚ ਪੌਦੇ ਦੇ ਰੋਗ ਵਿਗਿਆਨੀ ਜਾਂ ਖੇਤੀਬਾੜੀ ਮਾਹਿਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ:',
        'symptoms_help_list': [
            'ਉਪਚਾਰ ਤੋਂ ਬਾਅਦ ਵੀ ਲੱਛਣ ਰਹਿ ਜਾਣ',
            'ਰੋਗ ਤੇਜ਼ੀ ਨਾਲ ਫੈਲ ਰਿਹਾ ਹੋਵੇ',
            'ਕਈ ਪੌਦੇ ਪ੍ਰਭਾਵਿਤ ਹੋਣ',
            'ਅਜਿਹੇ ਲੱਛਣ ਆਉਣ',
        ],
        'resources_additional_title': 'ਵਧੀਕ ਸਰੋਤ',
        'resources_research': 'ਰੋਗ ਪ੍ਰਬੰਧਨ ਉੱਤੇ ਵਿਗਿਆਨਕ லிக்கியங்களை அணுகவும்',
        'resources_expert': 'விவசாய நிபுணர்களை அணுகவும்',
        'resources_community': 'ஆலோசனைக்காக விவசாய சமுதாயங்களைச் சேரவும்',
    },
    'ta': {
        'title': 'தாவர நோய் அறிவியல்',
        'subtitle': 'ஏஐ இயக்கும் பயிர் நோய் கண்டறிதல்',
        'upload_label': 'தாவர இலை படத்தை பதிவேற்றவும்',
        'predict_btn': 'முன்னறிவு',
        'disease': 'நோய்',
        'desc': 'விவரம்',
        'stage': 'எப்போது ஏற்படுகிறது',
        'show_accuracy': 'துல்லியத்தை காட்டு',
        'accuracy': 'துல்லியம்',
        'download_report': 'அறிக்கை பதிவிறக்கவும் (PDF)',
        'share': 'பகிர்',
        'recent_predictions': 'சமீபத்திய கணிப்புகள்',
        'no_predictions': 'இன்னும் கணிப்புகள் இல்லை. தொடங்க படத்தை பதிவேற்றவும்.',
        'current_session': 'தற்போதைய அமர்வு',
        'confidence_text': 'துல்லியம்',
        'tab_prevention': 'முன்னெச்சரிக்கை',
        'tab_treatment': 'சிகிச்சை',
        'tab_symptoms': 'அறிகுறிகள்',
        'tab_resources': 'வளங்கள்',
    }
}

if __name__ == '__main__':
    app.run(debug=True)
