# -*- coding: utf-8 -*-
"""
Generador de PDF profesional - Informe Tecnico SCY1101 EP2
Autor del proyecto: Matias Retamal
"""

import os, re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Preformatted, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus.flowables import BalancedColumns

# ── RUTAS ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MD   = os.path.join(BASE, "INFORME_TECNICO.md")
OUT  = os.path.join(BASE, "INFORME_TECNICO_EP2.pdf")

W, H  = A4
MAR   = 2.4*cm
INNER = W - 2*MAR

# ── PALETA ──────────────────────────────────────────────────────────────────
C_NAVY    = colors.HexColor("#0f2d5c")
C_BLUE    = colors.HexColor("#1d6fcd")
C_LBLUE   = colors.HexColor("#e8f1fc")
C_TEAL    = colors.HexColor("#0e7490")
C_SLATE   = colors.HexColor("#475569")
C_LGRAY   = colors.HexColor("#f1f5f9")
C_DGRAY   = colors.HexColor("#64748b")
C_BORDER  = colors.HexColor("#cbd5e1")
C_WHITE   = colors.white
C_BLACK   = colors.HexColor("#0f172a")
C_GREEN   = colors.HexColor("#15803d")
C_ORANGE  = colors.HexColor("#b45309")
C_CODE_BG = colors.HexColor("#f8fafc")
C_CODE_FG = colors.HexColor("#1e293b")

# ── ESTILOS ─────────────────────────────────────────────────────────────────
def sty(name, **kw):
    base = kw.pop("parent", "Normal")
    return ParagraphStyle(name, parent=getSampleStyleSheet()[base], **kw)

# Portada
S_COVER_TITLE = sty("CoverTitle",
    fontSize=30, fontName="Helvetica-Bold", textColor=C_WHITE,
    alignment=TA_CENTER, leading=38, spaceAfter=6)
S_COVER_SUB = sty("CoverSub",
    fontSize=14, fontName="Helvetica", textColor=colors.HexColor("#bfdbfe"),
    alignment=TA_CENTER, leading=20, spaceAfter=4)
S_COVER_META = sty("CoverMeta",
    fontSize=10, fontName="Helvetica", textColor=colors.HexColor("#93c5fd"),
    alignment=TA_CENTER, leading=16, spaceAfter=3)

# Cuerpo
S_H1 = sty("H1",
    fontSize=16, fontName="Helvetica-Bold", textColor=C_WHITE,
    spaceAfter=0, spaceBefore=16, leading=22)
S_H2 = sty("H2",
    fontSize=13, fontName="Helvetica-Bold", textColor=C_NAVY,
    spaceAfter=3, spaceBefore=18, leading=18)
S_H3 = sty("H3",
    fontSize=11, fontName="Helvetica-Bold", textColor=C_BLUE,
    spaceAfter=4, spaceBefore=12, leading=15)
S_BODY = sty("Body",
    fontSize=9.5, textColor=C_BLACK,
    spaceAfter=6, spaceBefore=2, leading=15, alignment=TA_JUSTIFY)
S_BULLET = sty("Bullet",
    fontSize=9.5, textColor=C_BLACK,
    spaceAfter=4, spaceBefore=1, leading=14,
    leftIndent=16, firstLineIndent=-12)
S_SUB_BULLET = sty("SubBullet",
    fontSize=9.2, textColor=C_SLATE,
    spaceAfter=3, spaceBefore=1, leading=13,
    leftIndent=30, firstLineIndent=-12)
S_CODE = sty("Code",
    fontSize=7.8, fontName="Courier", textColor=C_CODE_FG,
    backColor=C_CODE_BG,
    spaceAfter=6, spaceBefore=4, leading=11.5,
    leftIndent=10, rightIndent=10,
    borderPadding=(5, 8, 5, 8))
S_TH = sty("TH",
    fontSize=8.5, fontName="Helvetica-Bold", textColor=C_WHITE,
    alignment=TA_CENTER, leading=12)
S_TD = sty("TD",
    fontSize=8.5, textColor=C_BLACK,
    alignment=TA_LEFT, leading=12)
S_TD_C = sty("TDC",
    fontSize=8.5, textColor=C_BLACK,
    alignment=TA_CENTER, leading=12)
S_CAPTION = sty("Caption",
    fontSize=8, textColor=C_DGRAY, alignment=TA_CENTER,
    spaceAfter=8, spaceBefore=2, fontName="Helvetica-Oblique")
S_FOOTER = sty("Footer",
    fontSize=7, textColor=C_DGRAY, alignment=TA_CENTER)

# ── HELPERS ─────────────────────────────────────────────────────────────────
EMOJI = {"✅":"[OK]","⚠️":"[!]","❌":"[X]","🚀":"[+]","📊":"[g]","📄":"[doc]","🎯":"[>]"}

def clean(text):
    for e,r in EMOJI.items():
        text = text.replace(e, r)
    text = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', text)
    text = re.sub(r'`([^`]+)`',
        r'<font face="Courier" size="8" color="#1e293b">\1</font>', text)
    return text

def h1_block(text):
    """Crea un bloque de encabezado H1 con fondo navy y margen negativo visual."""
    label = Paragraph(clean(text), S_H1)
    t = Table([[label]], colWidths=[INNER + 0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("LINEBELOW",     (0,0), (-1,-1), 2, C_BLUE),
    ]))
    return [Spacer(1, 10), t, Spacer(1, 8)]

def h2_block(text):
    p = Paragraph(clean(text), S_H2)
    hr = HRFlowable(width="100%", thickness=1.5, color=C_NAVY,
                    spaceAfter=6, spaceBefore=0)
    return [Spacer(1, 4), p, hr]

def h3_block(text):
    p = Paragraph(clean(text), S_H3)
    hr = HRFlowable(width="40%", thickness=0.6, color=C_BLUE,
                    hAlign="LEFT", spaceAfter=4, spaceBefore=0)
    return [p, hr]

def divider():
    return [
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=0.5, color=C_BORDER,
                   spaceAfter=4, spaceBefore=4),
    ]

def build_table(rows):
    data, sep_idx = [], []
    for i, row in enumerate(rows):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if re.match(r'^[\s\|\-:]+$', row):
            sep_idx.append(i)
            continue
        if not data:
            data.append([Paragraph(clean(c), S_TH) for c in cells])
        else:
            # detect numeric cells → center
            row_data = []
            for c in cells:
                s = S_TD_C if re.match(r'^[\d\.\-\+%✅❌⚠️\[\]OK!Xok ]+$', c.strip()) else S_TD
                row_data.append(Paragraph(clean(c), s))
            data.append(row_data)

    if len(data) < 2:
        return None
    ncols = len(data[0])
    cw = [INNER / ncols] * ncols
    t = Table(data, colWidths=cw, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_LGRAY, C_WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.35, C_BORDER),
        ("LINEBELOW",     (0, 0), (-1, 0),  1.5, C_BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t

# ── PARSER MARKDOWN ──────────────────────────────────────────────────────────
def parse(md_text):
    story = []
    lines = md_text.splitlines()
    i = 0
    in_code, code_buf = False, []
    in_table, tbl_buf = False, []

    def flush_table():
        nonlocal in_table, tbl_buf
        if tbl_buf:
            t = build_table(tbl_buf)
            if t:
                story.extend([Spacer(1,4), t, Spacer(1,8)])
        in_table, tbl_buf = False, []

    def flush_code():
        nonlocal in_code, code_buf
        if code_buf:
            story.append(Preformatted("\n".join(code_buf), S_CODE))
            story.append(Spacer(1,4))
        in_code, code_buf = False, []

    while i < len(lines):
        ln = lines[i]

        # ─ code fence
        if ln.strip().startswith("```"):
            if in_table: flush_table()
            if in_code:  flush_code()
            else:         in_code = True
            i += 1; continue
        if in_code:
            code_buf.append(ln); i += 1; continue

        # ─ table
        if ln.strip().startswith("|"):
            if not in_table: in_table = True; tbl_buf = []
            tbl_buf.append(ln); i += 1; continue
        else:
            if in_table: flush_table()

        # ─ HR
        if re.match(r'^-{3,}\s*$', ln.strip()):
            story.extend(divider()); i += 1; continue

        # ─ H1
        if re.match(r'^# [^#]', ln):
            story.extend(h1_block(ln[2:].strip())); i += 1; continue

        # ─ H2
        if re.match(r'^## [^#]', ln):
            story.extend(h2_block(ln[3:].strip())); i += 1; continue

        # ─ H3
        if re.match(r'^### ', ln):
            story.extend(h3_block(ln[4:].strip())); i += 1; continue

        # ─ sub-bullet (indent)
        if re.match(r'^\s{3,}[-*]\s', ln):
            txt = clean(re.sub(r'^\s+[-*]\s', '', ln))
            story.append(Paragraph(f"◦ {txt}", S_SUB_BULLET)); i += 1; continue

        # ─ bullet
        if re.match(r'^[-*]\s', ln):
            txt = clean(ln[2:].strip())
            story.append(Paragraph(f"• {txt}", S_BULLET)); i += 1; continue

        # ─ ordered list  (1. 2. …)
        if re.match(r'^\d+\.\s', ln):
            m = re.match(r'^(\d+)\.\s+(.*)', ln)
            if m:
                txt = clean(m.group(2))
                story.append(Paragraph(f"{m.group(1)}.  {txt}", S_BULLET))
            i += 1; continue

        # ─ blank
        if ln.strip() == "":
            story.append(Spacer(1,5)); i += 1; continue

        # ─ normal paragraph
        txt = clean(ln.strip())
        if txt:
            story.append(Paragraph(txt, S_BODY))
        i += 1

    if in_code:  flush_code()
    if in_table: flush_table()
    return story

# ── ENCABEZADO / PIE ─────────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()

    # ── header
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, H - 1.6*cm, W, 1.6*cm, fill=1, stroke=0)
    # accent line
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, H - 1.6*cm, W, 2, fill=1, stroke=0)

    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(MAR, H - 1.0*cm, "SCY1101 — Programación para la Ciencia de Datos")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MAR, H - 1.35*cm, "Pipeline de ML para Logística y Transporte  |  Matias Retamal")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(W - MAR, H - 1.1*cm, "Mayo 2026")

    # ── footer
    canvas.setFillColor(C_LGRAY)
    canvas.rect(0, 0, W, 1.1*cm, fill=1, stroke=0)
    canvas.setFillColor(C_BORDER)
    canvas.rect(0, 1.1*cm, W, 0.5, fill=1, stroke=0)

    canvas.setFillColor(C_SLATE)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(MAR, 0.42*cm,
        "github.com/Tg20taps/Prueba_progra")
    canvas.drawCentredString(W/2, 0.42*cm,
        f"Página  {doc.page}")
    canvas.drawRightString(W - MAR, 0.42*cm,
        "Kedro 1.3.1  |  Scikit-learn  |  26 nodos  |  30 modelos")
    canvas.restoreState()

# ── PORTADA ──────────────────────────────────────────────────────────────────
def cover_page():
    story = []

    # fondo navy completo simulado con un bloque de tabla
    header_data = [[Paragraph(
        "SCY1101 — Programación para la Ciencia de Datos",
        sty("CH", fontSize=10, fontName="Helvetica", textColor=colors.HexColor("#93c5fd"),
            alignment=TA_CENTER))]]
    ht = Table(header_data, colWidths=[INNER])
    ht.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), C_NAVY),
        ("TOPPADDING",  (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1),10),
    ]))
    story.append(ht)
    story.append(Spacer(1, 2.5*cm))

    # Título principal
    story.append(Paragraph("Informe Técnico", sty("CT1",
        fontSize=32, fontName="Helvetica-Bold", textColor=C_NAVY,
        alignment=TA_CENTER, spaceAfter=4)))
    story.append(Paragraph(
        "Pipeline de Machine Learning para Logística y Transporte",
        sty("CT2", fontSize=15, fontName="Helvetica", textColor=C_BLUE,
            alignment=TA_CENTER, spaceAfter=6, leading=22)))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="50%", thickness=2.5, color=C_NAVY,
                             hAlign="CENTER", spaceAfter=0.6*cm))

    # Datos del autor
    meta_rows = [
        ["Autor",      "Matias Retamal"],
        ["Asignatura", "SCY1101 — Programación para la Ciencia de Datos"],
        ["Evaluación", "Parcial N° 2"],
        ["Fecha",      "Mayo 2026"],
        ["Repositorio","github.com/Tg20taps/Prueba_progra"],
    ]
    meta_data = [[
        Paragraph(f"<b>{r[0]}</b>", sty("MK", fontSize=9, fontName="Helvetica-Bold",
            textColor=C_SLATE, alignment=TA_RIGHT)),
        Paragraph(r[1], sty("MV", fontSize=9, fontName="Helvetica",
            textColor=C_BLACK, alignment=TA_LEFT)),
    ] for r in meta_rows]
    mt = Table(meta_data, colWidths=[4*cm, INNER-4*cm], hAlign="CENTER")
    mt.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("LINEBELOW",     (0,0),(-1,-2), 0.3, C_BORDER),
    ]))
    story.append(mt)
    story.append(Spacer(1, 1.0*cm))

    # Tabla de resultados clave
    story.append(Paragraph("Resultados Clave del Proyecto",
        sty("RK", fontSize=11, fontName="Helvetica-Bold", textColor=C_NAVY,
            alignment=TA_CENTER, spaceAfter=6)))

    kpi_data = [
        [Paragraph("<b>Clasificación</b>", S_TH),
         Paragraph("<b>Regresión</b>",     S_TH),
         Paragraph("<b>Clustering</b>",    S_TH),
         Paragraph("<b>Pipeline</b>",      S_TH)],
        [Paragraph("GaussianNB\nF1 = 0.2963\nRecall = 96.6%",
                   sty("K2", fontSize=8.5, alignment=TA_CENTER, leading=13)),
         Paragraph("KNeighborsRegressor\nMAE = 1.44 días\nR² = 0.115",
                   sty("K3", fontSize=8.5, alignment=TA_CENTER, leading=13)),
         Paragraph("KMeans k=5\nSilhouette = 0.082\nDBSCAN / PCA",
                   sty("K4", fontSize=8.5, alignment=TA_CENTER, leading=13)),
         Paragraph("Kedro 1.3.1\n26 nodos · ~30 seg\n30 modelos evaluados",
                   sty("K5", fontSize=8.5, alignment=TA_CENTER, leading=13))],
    ]
    kpi_t = Table(kpi_data, colWidths=[INNER/4]*4)
    kpi_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1, 0), C_NAVY),
        ("BACKGROUND",    (0,1),(-1,-1), C_LBLUE),
        ("GRID",          (0,0),(-1,-1), 0.5, C_BORDER),
        ("LINEBELOW",     (0,0),(-1, 0), 2,   C_BLUE),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(kpi_t)
    story.append(Spacer(1, 0.8*cm))

    # Métodos obligatorios satisfechos
    req_data = [
        [Paragraph("<b>Requisito de Rúbrica</b>", S_TH),
         Paragraph("<b>Método</b>",               S_TH),
         Paragraph("<b>Estado</b>",               S_TH)],
        [Paragraph("IEE 2.1.1 — Modelos supervisados", S_TD),
         Paragraph("15 clasificadores + 15 regresores con CV-5", S_TD),
         Paragraph("[OK] CUMPLIDO", sty("OK", fontSize=8.5, textColor=C_GREEN,
             fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Paragraph("IEE 2.3.1 — Optimización", S_TD),
         Paragraph("GridSearchCV + RandomizedSearchCV", S_TD),
         Paragraph("[OK] CUMPLIDO", sty("OK2", fontSize=8.5, textColor=C_GREEN,
             fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Paragraph("IEE 2.1.2 — No supervisado", S_TD),
         Paragraph("KMeans · DBSCAN · PCA", S_TD),
         Paragraph("[OK] CUMPLIDO", sty("OK3", fontSize=8.5, textColor=C_GREEN,
             fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Paragraph("BONUS — Optuna (TPE)", S_TD),
         Paragraph("50 trials · bayesiano · comparado vs GridSearch", S_TD),
         Paragraph("[+] EXTRA", sty("EX", fontSize=8.5, textColor=C_BLUE,
             fontName="Helvetica-Bold", alignment=TA_CENTER))],
    ]
    req_t = Table(req_data, colWidths=[INNER*0.38, INNER*0.44, INNER*0.18])
    req_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1, 0), C_TEAL),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_LGRAY, C_WHITE]),
        ("GRID",          (0,0),(-1,-1), 0.35, C_BORDER),
        ("LINEBELOW",     (0,0),(-1, 0), 1.5, C_TEAL),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("RIGHTPADDING",  (0,0),(-1,-1), 7),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(req_t)

    story.append(PageBreak())
    return story

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    with open(MD, encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        OUT,
        pagesize=A4,
        leftMargin=MAR, rightMargin=MAR,
        topMargin=2.1*cm, bottomMargin=1.8*cm,
        title="Informe Técnico EP2 — SCY1101",
        author="Matias Retamal",
        subject="Pipeline ML Logística y Transporte",
        creator="Kedro + Scikit-learn",
    )

    story = cover_page() + parse(md_text)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    sz = os.path.getsize(OUT) / 1024
    print(f"\n[OK] PDF generado: {OUT}")
    print(f"     Tamaño: {sz:.1f} KB")

if __name__ == "__main__":
    main()
