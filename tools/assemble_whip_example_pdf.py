"""ReportLab page composition with embedded vector Matplotlib plots."""
from pathlib import Path
import io,json,shutil,hashlib
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.colors import HexColor
from pypdf import PdfReader,PdfWriter,Transformation
ROOT=Path(__file__).resolve().parents[1]
TMP=ROOT/'tmp/pdfs/M1_take004_whip';OUT=ROOT/'output/pdf'
d=json.loads((TMP/'figure_data.json').read_text())
W,H=landscape(A4);stream=io.BytesIO();c=canvas.Canvas(stream,pagesize=(W,H))
ink=HexColor('#213146');muted=HexColor('#667589');blue=HexColor('#2764b4');orange=HexColor('#d96728')
c.setFillColor(ink);c.setFont('Helvetica-Bold',18)
c.drawString(35,H-32,'Aerial whip: real flight vs. preflight prediction')
c.setFillColor(muted);c.setFont('Helvetica',9.2)
c.drawString(35,H-49,'M1-full  |  Real-flight take 004  |  MPPI-generated open-loop PVA  |  1.20 s whip')
# Shared trajectory legend at the upper right below the title.
y=H-69
c.setStrokeColor(orange);c.setLineWidth(1.8);c.line(35,y,56,y)
c.setFillColor(ink);c.setFont('Helvetica',8.8);c.drawString(62,y-3,'Measured flight')
c.setStrokeColor(blue);c.setDash(4,2);c.line(158,y,179,y);c.setDash();c.drawString(185,y-3,'Original M1 forecast')
c.setFont('Helvetica-Bold',9)
c.drawRightString(W-35,y-3,f'Tip RMS {d["tip_rms_cm"]:.2f} cm   |   Drone RMS {d["drone_rms_cm"]:.2f} cm   |   Tip coverage 100%')
# Light separator, plotting area and concise scientific qualifications.
c.setStrokeColor(HexColor('#dce2e9'));c.setLineWidth(.5);c.line(35,H-79,W-35,H-79)
c.setFillColor(ink);c.setFont('Helvetica',8)
c.drawString(35,49,f'Closest tip approach: {d["nearest_cm"]:.2f} cm from target centre at {d["nearest_time_s"]:.3f} s; no observed entry into the 5 cm sphere.')
c.setFillColor(muted);c.setFont('Helvetica',7.4)
c.drawString(35,36,'Residual = measured - predicted position, not the learned neural correction. Raw XYZ; estimated clock offset; no trajectory recentering.')
c.drawString(35,25,'Selected best-agreement example among 13 development takes, not aggregate validation. Marker gaps are retained; XZ shapes omit lateral error.')
c.drawString(35,14,'Source: whip_m1_004  |  Original plan 20260910-181929-716218  |  Prediction frozen before flight; no refit or regenerated forecast.')
c.showPage();c.save();stream.seek(0)
base=PdfReader(stream).pages[0];plot=PdfReader(TMP/'plots.pdf').pages[0]
# Leave the ReportLab heading and footer untouched. Keep vector paths/fonts.
plot_height=H-79-62;plot_width=W
scale=min(plot_width/float(plot.mediabox.width),plot_height/float(plot.mediabox.height))
dx=(W-float(plot.mediabox.width)*scale)/2
base.merge_transformed_page(plot,Transformation().scale(scale).translate(dx,62))
writer=PdfWriter();writer.add_page(base)
writer.add_metadata({'/Title':'M1 take 004 - Measured aerial whip and original prediction',
    '/Subject':'Selected development flight: cable trajectories and position residuals',
    '/Creator':'ReportLab and Matplotlib; evidence-bound project export'})
path=OUT/'M1_take004_whip_comparison.pdf'
with path.open('wb') as f:writer.write(f)
check=PdfReader(path);assert len(check.pages)==1
text=check.pages[0].extract_text()
for required in ('5.45','5.94','5.87','position residual','Selected best-agreement'):
    assert required.lower() in text.lower(),required
for p,h in d['protected_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
shutil.copy2(TMP/'figure_data.json',OUT/'M1_take004_whip_comparison_provenance.json')
print('Verified one-page PDF and source hashes:',path)
