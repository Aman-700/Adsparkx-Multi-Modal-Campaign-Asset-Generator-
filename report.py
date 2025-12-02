import csv, os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

OUT = os.getenv("OUT_DIR","./outputs")
os.makedirs(OUT, exist_ok=True)

class ReportGenerator:
    def __init__(self):
        pass

    def generate_report(self, brief: dict, assets: list):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        csv_path = os.path.join(OUT, f"report_{ts}.csv")
        pdf_path = os.path.join(OUT, f"report_{ts}.pdf")
        # write csv
        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product","copy","image_url","score"])
            for a in assets:
                writer.writerow([brief.get("product"), a["copy"], a["image_url"], a["score"]])
        # write simple pdf
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(40, 750, f"Campaign Report for {brief.get('product')} - {ts}")
        y = 720
        for a in assets:
            text = f"Score: {a['score']:.3f} | Copy: {a['copy']}"
            c.drawString(40, y, text)
            y -= 40
            if y < 100:
                c.showPage()
                y = 750
        c.save()
        return {"csv": csv_path, "pdf": pdf_path}
