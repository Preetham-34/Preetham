import io
import pdfplumber
import docx2txt
import magic

def parse_resume(file):
    """Extract text from PDF or DOCX files"""
    content = file.read()
    mime = magic.from_buffer(content, mime=True)
    
    if mime == "application/pdf":
        return parse_pdf(content)
    elif mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                 "application/msword"]:
        return parse_docx(content)
    else:
        raise ValueError(f"Unsupported file type: {mime}")

def parse_pdf(content):
    """Extract text from PDF using pdfplumber"""
    text = ""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def parse_docx(content):
    """Extract text from DOCX using docx2txt"""
    return docx2txt.process(io.BytesIO(content))
