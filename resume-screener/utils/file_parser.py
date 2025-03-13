import io
import PyPDF2
from docx import Document

def parse_resume(file):
    """Extract text from PDF or DOCX files"""
    content = file.read()
    
    if file.type == "application/pdf":
        return parse_pdf(content)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return parse_docx(content)
    else:
        raise ValueError("Unsupported file format")

def parse_pdf(content):
    """Extract text from PDF"""
    pdf = PyPDF2.PdfReader(io.BytesIO(content))
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def parse_docx(content):
    """Extract text from DOCX"""
    doc = Document(io.BytesIO(content))
    return "\n".join([para.text for para in doc.paragraphs])
