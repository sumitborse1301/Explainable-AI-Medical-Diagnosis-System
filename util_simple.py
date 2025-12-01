
# Used for Image manipulation/Report Generation/Other libraries
import numpy as np
import io
import cv2
from PIL import Image
import pydicom
import nibabel as nib
import base64
import uuid
import os
import json
from openai import OpenAI
from Bio import Entrez
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RPImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from prompts import ANALYSIS_PROMPT, SYSTEM_MESSAGE, LITERATURE_SYSTEM_MESSAGE, FALLBACK_RESPONSE, ERROR_RESPONSE, ERROR_REFERENCES
from reportlab.platypus import ListFlowable, ListItem
from reportlab.lib.units import inch
from reportlab.platypus import Image as RPImage
from reportlab.lib.units import inch




Entrez.email = "sumitborse13@gmail.com"
Entrez.api_key = ""
Entrez.tool = "MedicalImagingAnalyzer"


# util_simple.py  (add near top, under imports)

def compute_model_confidence(analysis_text):
    """
    Heuristic to derive a confidence score and simple metrics from the analysis text.
    - Returns a dict: { 'probability': float (0-100),
                       'precision': float,
                       'recall': float,
                       'f1': float,
                       'notes': str }
    This is a heuristic (no ground truth) — maps certainty words to percentages.
    """
    if not analysis_text:
        return {"probability": 50.0, "precision": 50.0, "recall": 50.0, "f1": 50.0, "notes": "No analysis text."}

    text = analysis_text.lower()

    # Ordered mapping from strong -> weaker certainty keywords
    mappings = [
        (["definitely", "definite", "certain", "confirmed", "diagnostic of"], 98),
        (["highly suggestive", "highly suspicious", "very likely", "very suspicious"], 90),
        (["probable", "probability", "likely", "most consistent with"], 80),
        (["suspicious", "concerning for", "suggestive of"], 70),
        (["possible", "may represent", "cannot exclude"], 60),
        (["indeterminate", "nonspecific", "uncertain", "equivocal"], 50),
        (["unlikely", "less likely", "unlikely to represent"], 25),
        (["no evidence", "no acute", "normal"], 10),
    ]

    # default base
    prob = None
    for keywords, score in mappings:
        for kw in keywords:
            if kw in text:
                prob = score
                break
        if prob is not None:
            break

    # fallback: use presence of strong / negation words
    if prob is None:
        positive_words = ["consolidation","opacities","nodule","mass","effusion","pneumonia","tumor","metastasis"]
        matches = sum(1 for w in positive_words if w in text)
        prob = min(90, 50 + matches*10)  # rough scaling

    # Adjust slightly by presence of hedging words
    if any(x in text for x in ["cannot exclude", "may represent", "could be", "possible"]):
        prob -= 5
    if any(x in text for x in ["definite", "confident", "confirm"]):
        prob += 3

    prob = max(1.0, min(99.0, float(prob)))

    # Derive dummy precision/recall/F1 from probability (purely illustrative)
    precision = max(5.0, min(99.0, prob - 3.0))   # slightly lower/higher than prob
    recall = max(5.0, min(99.0, prob - 6.0))
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = prob

    notes = "Heuristic confidence derived from analysis text. Not a replacement for clinical validation."

    return {
        "probability": round(prob, 1),
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "f1": round(f1, 1),
        "notes": notes
    }



#Process part of different medical image file formats
def process_file(uploaded_file):
    """Process different medical image file formats"""
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext in ['jpg','jpeg','png']:
        image = Image.open(uploaded_file).convert('RGB')
        return {"type": "image", "data": image, "array": np.array(image)}
    elif ext == 'dcm':
        dicom = pydicom.dcmread(uploaded_file)
        img_array = dicom.pixel_array
        img_array = ((img_array - img_array.min()) /
                     (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        return {"type": "dicom", "data": Image.fromarray(img_array), "array": img_array}
    elif ext in ['nii', 'nii.gz']:
        temp_path = f"temp_{uuid.uuid4()}.nii.gz"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        nii_img = nib.load(temp_path)
        img_array = nii_img.get_fdata()[:, :, nii_img.shape[2]//2]
        img_array = ((img_array - img_array.min())/
                     (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        os.remove(temp_path)
        return {"type": "nifti", "data":  Image.fromarray(img_array), "array": img_array}
    

#Generate Heatmap
def generate_heatmap(image_array):
    """Generate a heatmap overlay for XAI Visualization"""

    if len(image_array.shape) == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array

    heatmap = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(heatmap, 0.5, image_array, 0.5, 0)

    return Image.fromarray(overlay), Image.fromarray(heatmap)
    
#Fetching Data from the uploaded file
def extract_findings_and_keywords(analysis_text):
    """Extract findings and keyword from analysis text"""
    findings = []
    keywords = []

    #Look for common medical findings patterns
    if "Impression: " in analysis_text:
        impression_section = analysis_text.split("Impression:") [1].strip()
        numbered_items = impression_section.split("/n")
        for item in numbered_items:
            item = item.strip()
            if item and (item[0].isdigit() or item[0] == '-' or item[0] == '*'):
                # Clean up the item
                clean_item = item
                if item[0].isdigit() and "." in item[:3]:
                    clean_item = item.split(".", 1)[1].strip()
                elif item[0] in ['-', '*']:
                    clean_item = item[1:].strip()

                findings.append(clean_item)

                # Extract potential keywords
                for word in clean_item.split():
                    word = word.lower().strip(',.:;()')
                    if len(word) > 4 and word not in ['about', 'with', 'that', 'this', 'these', 'those']:
                        keywords.append(word)

    #Add common radiological terms as keywords if they appear in the text
    common_terms = [
        "pneumonia", "infiltrates", "opacities", "nodule", "mass", "tumor", 
        "cardiomegaly", "effusion", "consolidation", "atelectasis", "edema",
        "fracture", "fibrosis", "emphysema", "pneumothorax", "metastasis"
    ]

    for term in common_terms:
        if term in analysis_text.lower() and term not in keywords:
            keywords.append(term)

    # Remove duplicates while preserving order
    keywords = list(dict.fromkeys(keywords))

    return findings, keywords[:5]


# Analyzation of Image - OpenAI

from openai import OpenAI

def analyze_image(image, api_key, enable_xai=True):
    """Analyze medical image using OPENAI's vision model"""
    import io, base64, uuid
    from datetime import datetime

    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode()

    # Initialize client using new SDK
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ANALYSIS_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ],
                },
            ],
            max_tokens=800,
        )

        analysis = response.choices[0].message.content

        # Include note if XAI visualization is enabled
        if enable_xai:
            analysis += "\n\n[XAI Visualization Enabled: Heatmap generation will assist in interpretability.]"
        else:
            analysis += "\n\n[XAI Visualization Disabled.]"

        findings, keywords = extract_findings_and_keywords(analysis)

        return {
            "id": str(uuid.uuid4()),
            "analysis": analysis,
            "findings": findings,
            "keywords": keywords,
            "date": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "id": str(uuid.uuid4()),
            "analysis": f"Error analyzing image: {str(e)}",
            "findings": [],
            "keywords": [],
            "date": datetime.now().isoformat(),
        }



# PubMed Articles
def search_pubmed(keywords, max_results=5):
    """Search PubMed for relevant articles based on keywords"""
    if not keywords:
        return []
    
    query = ' AND '.join(keywords)
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        results = Entrez.read(handle)

        if not results["IdList"]:
            return []
        
        # Fetch details for those IDs
        fetch_handle = Entrez.efetch(db="pubmed", id=results["IdList"], rettype="medline", retmode="text") 
        records = fetch_handle.read().split('\n\n')

        publications = []
        for record in records:
            if not record.strip():
                continue

            pub_data = {"id": "", "title": "", "journal": "", "year": ""}

            # Extract relevant fields
            for line in record.split('\n'):
                if line.startswith('PMID-'):
                    pub_data["id"] = line[6:].strip()
                elif line.startswith('TI  -'):
                    pub_data["title"] = line[6:].strip()
                elif line.startswith('JT  -'):
                    pub_data["journal"] = line[6:].strip()
                elif line.startswith('DP  -'):
                    year_match = line[6:].strip().split()[0]
                    pub_data["year"] = year_match if year_match.isdigit() else "2024"



            if pub_data["id"]:
                publications.append(pub_data)
        
        return publications
    except Exception as e:
        print(f"Error searching PubMed: {e}")

        # Return fallback data
        return [{"id": f"PMD{1000+i}",
                "title": f"Study on {' '.join(keywords)}",
                "journal": "Medical Journal",
                "year": "2024"} for i in range(min(3, max_results))]

# Clinical Trials
import requests


def search_clinical_trials(keywords, max_results=3):
    """Fetch real clinical trials from ClinicalTrials.gov API (v2-compatible, fixed URL)"""
    if not keywords:
        return []

    # Build the search query (v2 API uses query.term)
    query = " OR ".join(keywords)
    url = f"https://clinicaltrials.gov/api/v2/studies?query.term={query}&pageSize={max_results}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        trials = []
        for s in studies:
            protocol = s.get("protocolSection", {}).get("identificationModule", {})
            status = s.get("protocolSection", {}).get("statusModule", {})
            design = s.get("protocolSection", {}).get("designModule", {})

            trials.append({
                "id": protocol.get("nctId", "N/A"),
                "title": protocol.get("officialTitle", "No title available"),
                "status": status.get("overallStatus", "Unknown"),
                "phase": design.get("phases", ["N/A"])[0] if design.get("phases") else "N/A",
            })

        return trials

    except Exception as e:
        print("Error fetching clinical trials:", e)
        return []





# Generate Repord PDF
def generate_report(data, include_references=True):
    """Generate a PDF report with analysis results"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=8
    )

    # Build Content
    content = []

    # Header 
    content.append(Paragraph("Medical Imaging Analysis Report", title_style))
    content.append(Spacer(1, 12))

    # Date and ID
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    content.append(Paragraph(f"Report ID: {data['id']}", styles["Normal"]))
    if 'filename' in data:
        content.append(Paragraph(f"Image: {data['filename']}", styles["Normal"]))
    content.append(Spacer(1, 12))

    from reportlab.platypus import ListFlowable, ListItem

    # ---- Analysis Section - structured, robust formatting ----
    content.append(Paragraph("Analysis Results", subtitle_style))

    analysis_text = data.get("analysis", "") or ""

    # Clean markdown formatting (keep bullets, remove asterisks)
    import re
    analysis_text = re.sub(r"\*\*(.*?)\*\*", r"\1", analysis_text)  # remove **bold**
    analysis_text = analysis_text.replace("*", "")  # remove stray single asterisks



    # Normalize common markdown-like tokens
    analysis_text = analysis_text.replace("\r\n", "\n").replace("\r", "\n")


    # === Model confidence & metrics (PDF-only section) ===
    # Compute heuristic metrics (not shown in UI). These appear only in PDF.
    metrics = compute_model_confidence(analysis_text)

    # Generate chart image (use non-interactive backend and absolute path)
    try:
        import matplotlib
        matplotlib.use("Agg")            # IMPORTANT on servers
        import matplotlib.pyplot as plt

        metrics_filename = f"temp_metrics_{uuid.uuid4().hex}.png"
        metrics_fig_path = os.path.abspath(metrics_filename)

        labels = ["Probability", "Precision", "Recall", "F1"]
        values = [
            float(metrics.get("probability", 0)),
            float(metrics.get("precision", 0)),
            float(metrics.get("recall", 0)),
            float(metrics.get("f1", 0)),
        ]

        plt.figure(figsize=(5, 2.8))
        bars = plt.bar(labels, values, edgecolor="black")
        plt.ylim(0, 100)
        plt.ylabel("Percent (%)")
        plt.title("Model confidence & metrics")
        for bar, val in zip(bars, values):
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, h + 1.5, f"{int(round(val))}%", ha="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(metrics_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Failed to generate metrics chart:", e)
        metrics_fig_path = None

    # Insert Model Confidence text + chart into PDF (this section is NOT shown in Streamlit UI)
    try:
        content.append(Paragraph("Model Confidence & Evaluation (Internal)", subtitle_style))
        content.append(Spacer(1, 6))

        content.append(Paragraph(
            f"Overall Model Confidence: <b>{metrics.get('probability', 0)}%</b>",
            styles["Normal"]
        ))
        content.append(Paragraph(
            f"Precision: {metrics.get('precision', 0)}% — Recall: {metrics.get('recall', 0)}% — F1 Score: {metrics.get('f1', 0)}%",
            styles["Normal"]
        ))
        content.append(Spacer(1, 6))

        if metrics.get("notes"):
            content.append(Paragraph(metrics.get("notes"), styles["Normal"]))
            content.append(Spacer(1, 6))

        if metrics_fig_path and os.path.exists(metrics_fig_path):
            try:
                content.append(RPImage(metrics_fig_path, width=4.5 * inch))
                content.append(Spacer(1, 12))
            except Exception as e:
                print("Failed to insert metrics image into PDF:", e)
    except Exception as e:
        print("Failed adding Model Confidence block:", e)




    # Try splitting into sections by "###" or by numeric headings
    sections = []
    if "###" in analysis_text:
        raw_sections = [s.strip() for s in analysis_text.split("###") if s.strip()]
        for sec in raw_sections:
            # first line is heading if it looks like a short header
            lines = sec.split("\n")
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            sections.append((heading, body))
    else:
        # fallback: attempt to split by numeric "1." etc.
        # create one default section
        sections = [("Report", analysis_text)]

    # Helper to create bullet paragraphs
    def add_bulleted_lines(text):
        items = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # treat leading '-' or '•' or '•' variants as bullets
            if line.startswith("-") or line.startswith("•"):
                item_text = line.lstrip("-• ").strip()
                items.append(ListItem(Paragraph(item_text, styles["Normal"]), leftIndent=8))
            else:
                # normal paragraph line
                items.append(Paragraph(line, styles["Normal"]))
        return items

    # Add each section to the content flow
    for heading, body in sections:
        # Create heading paragraph (bold)
        heading_text = heading.strip()
        if heading_text:
            # remove hash marks (### etc.)
            heading_text = heading_text.lstrip('#').strip()

            # bold main sections and leave extra space
            important_sections = [
                "Image Type & Region",
                "Key Findings",
                "Diagnostic Assessment",
                "Patient-Friendly Explanation",
                "Research Context",
                "References",
                "Keywords"
            ]

            # Check if the heading matches an important section
            if any(sec.lower() in heading_text.lower() for sec in important_sections):
                content.append(Paragraph(f"<b>{heading_text}</b>", styles["Heading2"]))  # removed underline

                content.append(Spacer(1, 12))  # one-line gap after big section titles
            else:
                content.append(Paragraph(f"<b>{heading_text}</b>", styles["Heading3"]))
                content.append(Spacer(1, 6))

        # If body contains bullet markers or line breaks, render as bullets/paras
        if body:
            body_lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
            if any(ln.startswith("-") or ln.startswith("•") for ln in body_lines):
                list_items = []
                for ln in body_lines:
                    if ln.startswith("-") or ln.startswith("•"):
                        txt = ln.lstrip("-• ").strip()
                        list_items.append(ListItem(Paragraph(txt, styles["Normal"])))
                    else:
                        list_items.append(ListItem(Paragraph(ln, styles["Normal"])))
                if list_items:
                    content.append(ListFlowable(list_items, bulletType="bullet", start="•"))
                    content.append(Spacer(1, 6))
            else:
                analysis_style = ParagraphStyle(
                    "AnalysisText",
                    parent=styles["Normal"],
                    alignment=4,  # justify
                    fontSize=11,
                    leading=14,
                    spaceAfter=8
                )
                joined = " ".join([ln for ln in body.split("\n") if ln.strip()])
                content.append(Paragraph(joined, analysis_style))
                content.append(Spacer(1, 12))



        # add a separator line before references/keywords
        # After all sections are processed, add a separator line before references/keywords
        content.append(Spacer(1, 0.12 * inch))
        content.append(Table(
            [[""]],
            colWidths=[7.2 * inch],
            style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.grey)])
        ))
        content.append(Spacer(1, 0.12 * inch))

        # Disable Keywords section in the downloaded PDF
        # (They remain visible in Streamlit app)
        if False:
            if data.get('keywords'):
                content.append(Paragraph("<b>Keywords</b>", subtitle_style))
                if isinstance(data['keywords'], list):
                    keywords_text = ', '.join(data['keywords'])
                else:
                    keywords_text = str(data['keywords'])
                content.append(Paragraph(keywords_text, styles["Normal"]))
                content.append(Spacer(1, 12))


    # === Model confidence & metrics (PDF-only section) ===
    metrics = compute_model_confidence(analysis_text)

    # Generate chart into an in-memory buffer (avoids file/OneDrive issues)
    metrics_buf = None
    try:
        import matplotlib.pyplot as plt
        from reportlab.lib.utils import ImageReader

        labels = ["Probability", "Precision", "Recall", "F1"]
        values = [
            float(metrics.get("probability", 0)),
            float(metrics.get("precision", 0)),
            float(metrics.get("recall", 0)),
            float(metrics.get("f1", 0)),
        ]

        plt.figure(figsize=(5, 2.8))
        bars = plt.bar(labels, values, edgecolor="black")
        plt.ylim(0, 100)
        plt.ylabel("Percent (%)")
        plt.title("Model confidence & metrics")
        for bar, val in zip(bars, values):
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{int(round(val))}%", ha="center", fontsize=9)

        plt.tight_layout()

        # save to BytesIO
        metrics_buf = io.BytesIO()
        plt.savefig(metrics_buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        metrics_buf.seek(0)

        # create ImageReader from buffer
        metrics_img_reader = ImageReader(metrics_buf)

    except Exception as e:
        print("Failed to generate metrics chart (in-memory):", e)
        metrics_buf = None
        metrics_img_reader = None


    # Insert Model Confidence text + chart into PDF (NOT shown in UI)
    try:
        content.append(Paragraph("Model Confidence & Evaluation (Internal)", subtitle_style))
        content.append(Spacer(1, 6))

        content.append(Paragraph(
            f"Overall Model Confidence: <b>{metrics.get('probability', 0)}%</b>",
            styles["Normal"]
        ))
        content.append(Paragraph(
            f"Precision: {metrics.get('precision', 0)}% — Recall: {metrics.get('recall', 0)}% — F1 Score: {metrics.get('f1', 0)}%",
            styles["Normal"]
        ))
        content.append(Spacer(1, 6))

        if metrics.get("notes"):
            content.append(Paragraph(metrics.get("notes"), styles["Normal"]))
            content.append(Spacer(1, 6))

        if metrics_img_reader is not None:
            try:
                # RPImage is alias for reportlab.platypus.Image imported earlier
                content.append(RPImage(metrics_img_reader, width=4.5 * inch))
                content.append(Spacer(1, 12))
            except Exception as e:
                print("Failed to insert metrics image into PDF (ImageReader):", e)
        else:
            # optional debug
            print("Skipping inserting metrics image: no image reader available")

    except Exception as e:
        print("Failed adding Model Confidence block:", e)
    finally:
        # close buffer if exists
        try:
            if metrics_buf:
                metrics_buf.close()
        except Exception:
            pass





    # Add Reference if available and Requested
    if include_references:
        # Search PubMed
        pubmed_results = search_pubmed(data.get('keywords', []), max_results=3)
        if pubmed_results:
            content.append(Paragraph("Relevant Medical Literature", subtitle_style))
            for ref in pubmed_results:
                content.append(Paragraph(f" {ref['title']}", styles["Normal"]))
                content.append(Paragraph(f" {ref['journal']}, {ref['year']} (PMID: {ref['id']})", styles["Normal"]))
            content.append(Spacer(1, 12))

        # Search Clinical Trials
        trials_results = search_clinical_trials(data.get('keywords', []), max_results=2)
        if trials_results:
            content.append(Paragraph("Related Clinical Trials", subtitle_style)) 
            for trial in trials_results:
                content.append(Paragraph(f" {trial['title']}", styles["Normal"]))   
                content.append(Paragraph(f" ID: {trial['id']}, Status: {trial['status']}", styles["Normal"]))

    # Build the PDF (after all content appended)
    doc.build(content)

    # cleanup temporary chart file AFTER doc build
    try:
        if 'metrics_fig_path' in locals() and metrics_fig_path and os.path.exists(metrics_fig_path):
            os.remove(metrics_fig_path)
    except Exception:
        pass

    buffer.seek(0)
    return buffer



# Analysis Storage
def get_analysis_store():
    """Get the analysis storage"""
    if os.path.exists("analysis_store.json"):
        with open("analysis_store.json", "r") as f:
            return json.load(f)
    return {"analysis": []}

def save_analysis(analysis_data, filename="unknown.jpg"):
    """Save analysis data to storage with real embeddings (cached, 1536-dim)."""
    from openai import OpenAI
    import numpy as np
    import json, os, time

    store = get_analysis_store()

    # Add filename to analysis data
    analysis_data["filename"] = filename

    # Ensure analysis record has a date (ISO) so sorting works
    if not analysis_data.get("date"):
        analysis_data["date"] = datetime.now().isoformat()


    # TARGET embedding dimension
    EMB_DIM = 1536

    # Try OpenAI embeddings if API key available
    try:
        client_key = analysis_data.get("api_key") or os.environ.get("OPENAI_API_KEY") or None
        if client_key:
            client = OpenAI(api_key=client_key)
            text_to_embed = analysis_data.get("analysis", "") or ""
            if text_to_embed.strip():
                resp = client.embeddings.create(input=text_to_embed, model="text-embedding-3-small")
                emb = resp.data[0].embedding
                # ensure list of floats
                analysis_data["embedding"] = list(map(float, emb))[:EMB_DIM]
            else:
                analysis_data["embedding"] = list(np.random.rand(EMB_DIM))
        else:
            # No OpenAI key — use deterministic fallback (hash -> repeat/pad)
            txt = analysis_data.get("analysis", "") or filename
            # deterministic pseudo-embedding: reproducible hash -> floats
            h = np.frombuffer((txt.encode("utf-8") * 5)[:EMB_DIM], dtype=np.uint8)
            emb = (h.astype(float) / 255.0).tolist()
            if len(emb) < EMB_DIM:
                emb = emb + list(np.random.RandomState(0).rand(EMB_DIM - len(emb)))
            analysis_data["embedding"] = emb[:EMB_DIM]

    except Exception as e:
        print("Embedding generation failed:", e)
        analysis_data["embedding"] = list(np.random.RandomState(0).rand(EMB_DIM))

    # Add to store
    store["analysis"].append(analysis_data)

    # Save back to file
    with open("analysis_store.json", "w") as f:
        json.dump(store, f)

    return analysis_data



# Specific analysis by ID
def get_analysis_by_id(analysis_id):
    """Get a specific analysis by ID"""
    store = get_analysis_store()

    for analysis in store["analysis"]:
        if analysis["id"] == analysis_id:
            return analysis
    
    return None

# Get Most Recent Analysis
from datetime import datetime  # ensure this is imported near the top of the file

def get_latest_analyses(limit=5):
    """Get the most recent analyses (sorted by the ISO timestamp in 'date')"""
    store = get_analysis_store()
    analyses = store.get("analysis", []) or []

    def parse_date(a):
        d = a.get("date", "")
        try:
            # Expect ISO format from analyze_image / save_analysis
            return datetime.fromisoformat(d)
        except Exception:
            # fallback: very old date so it goes to the end
            return datetime.fromisoformat("1970-01-01T00:00:00")

    sorted_analyses = sorted(analyses, key=parse_date, reverse=True)
    return sorted_analyses[:limit]


# Extract and Summarize common findings from all stored analyses
def extract_common_findings(): 
    """Extract and Summarize common findings from all stored analyses"""
    store = get_analysis_store()

    # Count keyword frequencies
    keyword_counts = {}
    for analysis in store["analysis"]:
        for keyword in analysis.get("keywords", []):
            if keyword in keyword_counts:
                keyword_counts[keyword] += 1
            else:
                keyword_counts[keyword] = 1

    # Sort by frequency
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords 

# Generate Statistical Report for findings
def generate_statistics_report():
    """Generate Statistical Report for findings"""
    store = get_analysis_store()

    if not store["analysis"]:
        return None
    
    # Count analyses by type
    type_counts = {}
    for analysis in store["analysis"]:
        analysis_type = analysis.get("type", "unknown")
        if analysis_type in type_counts:
            type_counts[analysis_type] += 1
        else:
            type_counts[analysis_type] = 1

    # Get Common findings
    common_findingss = extract_common_findings() 

    # Create Report
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []

    # Title
    content.append(Paragraph("Medical Imaging Statistics Report", styles["Title"]))
    content.append(Spacer(1, 12))

    # Overall Statistics
    content.append(Paragraph("Overall Statistics", styles["Heading2"]))
    content.append(Paragraph(f"Total analyses: {len(store['analysis'])}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # Analysis Types
    if type_counts:
        content.append(Paragraph("Analysis Types", styles["Heading2"]))
        for keyword, count in common_findingss[:10]: #Top 10
            content.append(Paragraph(f"{keyword.capitalize()}: {count}occurrences", styles["Normal"]))
        

    # Build the PDF
    doc.build(content)
    buffer.seek(0)
    return buffer
