import docx
import pdfplumber as pdfp
import easyocr


def load_docx(path):
    document = docx.Document(path)
    docx_text = ""
    for para in document.paragraphs:
        docx_text += para.text
    return docx_text


def load_pdf(path, skip_first: bool = True, join: bool = True):
    pdf = pdfp.open(path)
    # 用于存储整理后的文本内容
    pages_text = []
    for page in pdf.pages:
        page_text = []
        text = page.extract_text()
        # if text:
        #     lines = text.split('\n')
        #     page_text.extend(lines)
        # pages_text.append(page_text)
        pages_text.append(text)

    if skip_first:
        pages_text = pages_text[1:]

    result_text = "".join(pages_text)
    if join:
        return result_text
    else:
        return pages_text


def load_pic(path):
    pic_text = ""
    reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
    result = reader.readtext(path, detail=0)
    for text in result:
        pic_text += text
    return pic_text


