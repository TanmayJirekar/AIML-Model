import re
from PyPDF2 import PdfReader
from fpdf import FPDF
import pandas as pd

def parse_resume(file_obj):
    """
    Extracts text from a PDF file object and performs basic heuristic NLP 
    to extract years of experience, education_level, certifications count, and projects.
    """
    text = ""
    try:
        reader = PdfReader(file_obj)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None
    
    text_lower = text.lower()
    
    # Heuristics for Experience
    exp_matches = re.findall(r'(\d+)\+?\s*(years?|yrs?)(?:\s*of)?\s*experience', text_lower)
    exp = max([int(m[0]) for m in exp_matches]) if exp_matches else 2  # Default 2
    
    # Heuristics for Education Level (1: High School, 2: Bachelors, 3: Masters, 4: PhD)
    edu_level = 2 # default bachelors
    if any(word in text_lower for word in ['phd', 'doctorate', 'ph.d']):
        edu_level = 4
    elif any(word in text_lower for word in ['master', 'msc', 'mba', 'm.sc', 'm.tech', 'ms']):
        edu_level = 3
    elif any(word in text_lower for word in ['bachelor', 'bsc', 'b.tech', 'be', 'b.e.']):
        edu_level = 2
    elif any(word in text_lower for word in ['high school', 'diploma']):
        edu_level = 1
        
    # Certifications count heuristic
    cert_keywords = ['aws', 'azure', 'gcp', 'certified', 'certification', 'cfa', 'pmp', 'cisco', 'ccna', 'ccnp', 'coursera', 'udemy']
    cert_count = sum(1 for kw in cert_keywords if kw in text_lower)
    
    # Projects heuristic
    proj_keywords = ['project', 'developed', 'deployed', 'built', 'created', 'designed', 'analyzed']
    proj_count = int(sum(text_lower.count(kw) for kw in proj_keywords) / 3) # rough div to prevent over-counting
    if proj_count == 0 and 'project' in text_lower:
        proj_count = 2 # default
        
    return {
        "experience": min(exp, 50),
        "education_level": edu_level,
        "certifications": min(cert_count, 20),
        "projects": min(proj_count, 50)
    }

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI Career Intelligence Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(profile, predictions_df, advice_text):
    """
    Generates a PDF report saving it to bytes.
    predictions_df contains 1, 3, 5, 10 year predictions.
    """
    pdf = PDFReport()
    pdf.add_page()
    
    # Profile Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Candidate Profile:', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f"Experience:      {profile['experience']} years", 0, 1)
    pdf.cell(0, 10, f"Education Lvl:   {profile['education_level']}", 0, 1)
    pdf.cell(0, 10, f"Age:             {profile['age']} years", 0, 1)
    pdf.cell(0, 10, f"Certifications:  {profile['certifications']}", 0, 1)
    pdf.cell(0, 10, f"Projects:        {profile['projects']}", 0, 1)
    
    pdf.ln(5)
    
    # Predictions Table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Career Salary Trajectory:', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    for idx, row in predictions_df.iterrows():
        # Clean formatting
        pdf.cell(0, 10, f"Year {row['Year']}: Rs. {row['Salary']:,.2f}", 0, 1)
        
    pdf.ln(5)
    
    # AI Advice
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'AI Career Advice:', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 10, advice_text)
    
    return pdf.output(dest='S').encode('latin1')
