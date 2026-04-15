from reportlab.pdfgen import canvas
from pathlib import Path

def create_dummy_pdf(path: str):
    c = canvas.Canvas(path)
    c.drawString(100, 750, "Dummy Research Paper")
    c.drawString(100, 730, "The dataset used by the authors is the MNIST dataset.")
    c.drawString(100, 710, "Our methodology involves using a standard Multi-Layer Perceptron.")
    c.drawString(100, 690, "The accuracy achieved is 99%.")
    c.save()

if __name__ == "__main__":
    create_dummy_pdf("dummy.pdf")
    print("Created dummy.pdf")
