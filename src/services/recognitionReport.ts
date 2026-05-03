
import * as jsPDFModule from 'jspdf';
import type { PredictImageState } from '../state/workspaceState';

function resolveJsPDF(): any {
  const m: any = jsPDFModule;
  if (typeof m === 'function') return m;
  if (m?.jsPDF) return m.jsPDF;
  if (m?.default?.jsPDF) return m.default.jsPDF;
  if (typeof m?.default === 'function') return m.default;
  if (typeof (window as any).jspdf?.jsPDF === 'function') return (window as any).jspdf.jsPDF;
  throw new Error('Unable to resolve jsPDF constructor');
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

const CATEGORY_COLORS: Record<string, [number, number, number]> = {
  logo: [139, 92, 246],
  signature: [236, 72, 153],
  stamp: [245, 158, 11],
};

/**
 * Build an annotated image data URL — the original with bounding boxes
 * drawn over each located region, plus labels and rotation info.
 */
export async function buildAnnotatedImage(
  item: PredictImageState
): Promise<{ dataUrl: string; width: number; height: number } | null> {
  if (!item.prediction) return null;
  const img = await loadImage(item.dataUrl);
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  const regions = item.prediction.regions || [];

  // Only annotate real located regions — do NOT draw a full-image frame.
  for (let i = 0; i < regions.length; i++) {
    const r = regions[i];
    const color = CATEGORY_COLORS[r.category] || [99, 102, 241];
    const x = r.x * img.width;
    const y = r.y * img.height;
    const w = r.width * img.width;
    const h = r.height * img.height;

    const lw = Math.max(3, Math.min(img.width, img.height) * 0.006);
    ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},0.95)`;
    ctx.lineWidth = lw;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},0.12)`;
    ctx.fillRect(x, y, w, h);

    const badgeSize = Math.max(22, Math.min(img.width, img.height) * 0.035);
    ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
    ctx.beginPath();
    ctx.arc(x + badgeSize / 2, y + badgeSize / 2, badgeSize / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'white';
    ctx.font = `bold ${Math.round(badgeSize * 0.6)}px Inter, Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), x + badgeSize / 2, y + badgeSize / 2);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';

    let labelText = `${r.label} · ${(r.confidence * 100).toFixed(0)}%`;
    if (r.rotation !== undefined && r.rotation !== 0) {
      labelText += r.rotation === -1 ? ' · mirrored' : ` · ↻${r.rotation}°`;
    }
    drawLabel(ctx, x, y + h + 4, labelText, color);
  }

  return {
    dataUrl: canvas.toDataURL('image/jpeg', 0.9),
    width: img.width,
    height: img.height,
  };
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  color: [number, number, number]
) {
  const fontSize = Math.max(14, Math.min(ctx.canvas.width, ctx.canvas.height) * 0.022);
  ctx.font = `bold ${fontSize}px Inter, Arial, sans-serif`;
  const metrics = ctx.measureText(text);
  const pad = 6;
  const w = metrics.width + pad * 2;
  const h = fontSize + pad * 1.4;

  ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
  ctx.fillRect(x, y, w, h);

  ctx.fillStyle = 'white';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + pad, y + h / 2);
  ctx.textBaseline = 'alphabetic';
}

export async function exportRecognitionReport(
  items: PredictImageState[],
  filename = 'recognition_report.pdf',
  onProgress?: (stage: string, progress: number) => void
): Promise<void> {
  const done = items.filter((i) => i.status === 'done' && i.prediction && i.prediction.regions && i.prediction.regions.length > 0);
  if (done.length === 0) {
    throw new Error('No located regions to export.');
  }

  const JsPDFCtor = resolveJsPDF();
  const pdf = new JsPDFCtor({ unit: 'pt', format: 'a4' });
  const pageW = pdf.internal.pageSize.getWidth();
  const pageH = pdf.internal.pageSize.getHeight();
  const margin = 36;

  // ---- Cover page ----
  onProgress?.('Building cover…', 0.02);
  pdf.setFillColor(10, 14, 26);
  pdf.rect(0, 0, pageW, pageH, 'F');

  pdf.setTextColor(230, 233, 242);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(28);
  pdf.text('AI Recognition Report', margin, 120);

  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(13);
  pdf.setTextColor(168, 176, 199);
  pdf.text(`Generated: ${new Date().toLocaleString()}`, margin, 148);

  let totalRegions = 0;
  const categoryCounts: Record<string, number> = {};
  const labelCounts: Record<string, number> = {};
  for (const it of done) {
    const regs = it.prediction!.regions || [];
    totalRegions += regs.length;
    for (const r of regs) {
      categoryCounts[r.category] = (categoryCounts[r.category] || 0) + 1;
      labelCounts[r.label] = (labelCounts[r.label] || 0) + 1;
    }
  }

  pdf.text(`Images with matches: ${done.length}`, margin, 168);
  pdf.text(`Total elements located: ${totalRegions}`, margin, 188);

  let yCur = 230;
  pdf.setFontSize(11);
  pdf.setTextColor(168, 176, 199);
  pdf.text('Category breakdown:', margin, yCur);
  yCur += 20;
  pdf.setTextColor(230, 233, 242);
  pdf.setFontSize(12);
  Object.entries(categoryCounts).forEach(([cat, count]) => {
    const color = CATEGORY_COLORS[cat] || [99, 102, 241];
    pdf.setFillColor(color[0], color[1], color[2]);
    pdf.circle(margin + 6, yCur - 3, 4, 'F');
    pdf.text(`${cat}: ${count}`, margin + 18, yCur);
    yCur += 18;
  });

  yCur += 10;
  pdf.setTextColor(168, 176, 199);
  pdf.setFontSize(11);
  pdf.text('Unique labels identified:', margin, yCur);
  yCur += 20;
  pdf.setTextColor(230, 233, 242);
  pdf.setFontSize(12);
  Object.entries(labelCounts).sort((a, b) => b[1] - a[1]).slice(0, 10).forEach(([lab, count]) => {
    pdf.text(`• ${lab} (${count}×)`, margin + 6, yCur);
    yCur += 16;
  });

  pdf.setFontSize(10);
  pdf.setTextColor(110, 119, 151);
  pdf.text('Each subsequent page shows one scanned image with located elements highlighted.', margin, pageH - margin - 6);

  // ---- One page per image ----
  for (let i = 0; i < done.length; i++) {
    const item = done[i];
    onProgress?.(
      `Annotating image ${i + 1}/${done.length}`,
      0.05 + (0.9 * (i + 1)) / done.length
    );

    pdf.addPage();
    pdf.setFillColor(255, 255, 255);
    pdf.rect(0, 0, pageW, pageH, 'F');

    const annotated = await buildAnnotatedImage(item);
    if (!annotated) continue;

    const pred = item.prediction!;
    const regs = pred.regions || [];
    // Use category of the highest-confidence region for the header color
    const topReg = regs[0];
    const catColor = CATEGORY_COLORS[topReg?.category || pred.category] || [99, 102, 241];

    pdf.setFillColor(catColor[0], catColor[1], catColor[2]);
    pdf.rect(0, 0, pageW, 64, 'F');
    pdf.setTextColor(255, 255, 255);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(18);
    pdf.text(`${regs.length} element${regs.length === 1 ? '' : 's'} located`, margin, 30);
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(11);
    const uniqueLabels = Array.from(new Set(regs.map((r: any) => r.label))).join(', ');
    pdf.text(
      `${uniqueLabels}  ·  Image ${i + 1} of ${done.length}`,
      margin,
      50
    );

    pdf.setFontSize(9);
    pdf.setTextColor(100, 100, 120);
    pdf.setFont('helvetica', 'normal');
    pdf.text(`Source: ${item.sourceName}`, margin, 82);

    const imgAreaY = 96;
    const imgAreaH = pageH - imgAreaY - margin - 180;
    const imgAreaW = pageW - 2 * margin;

    const imgRatio = Math.min(
      imgAreaW / annotated.width,
      imgAreaH / annotated.height
    );
    const drawW = annotated.width * imgRatio;
    const drawH = annotated.height * imgRatio;
    const imgX = (pageW - drawW) / 2;
    const imgY = imgAreaY;

    pdf.setDrawColor(220, 220, 230);
    pdf.setLineWidth(1);
    pdf.rect(imgX - 2, imgY - 2, drawW + 4, drawH + 4);
    pdf.addImage(annotated.dataUrl, 'JPEG', imgX, imgY, drawW, drawH, undefined, 'FAST');

    const detailY = imgY + drawH + 20;
    pdf.setTextColor(30, 30, 40);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(12);
    pdf.text('Located Elements', margin, detailY);

    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(10);
    pdf.setTextColor(60, 60, 80);

    let ry = detailY + 18;
    regs.slice(0, 6).forEach((r: any, idx: number) => {
      const col = CATEGORY_COLORS[r.category] || [99, 102, 241];
      pdf.setFillColor(col[0], col[1], col[2]);
      pdf.circle(margin + 6, ry - 3, 5, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(8);
      pdf.text(String(idx + 1), margin + 6, ry - 1, { align: 'center' });

      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(10);
      pdf.setTextColor(30, 30, 40);
      const xPct = (r.x * 100).toFixed(0);
      const yPct = (r.y * 100).toFixed(0);
      const wPct = (r.width * 100).toFixed(0);
      const hPct = (r.height * 100).toFixed(0);
      let extras = '';
      if (r.rotation !== undefined && r.rotation !== 0) {
        extras = r.rotation === -1 ? ' · mirrored' : ` · rotated ${r.rotation}°`;
      }
      pdf.text(
        `${r.label} — ${(r.confidence * 100).toFixed(0)}% · region: ${xPct}%,${yPct}% (${wPct}%×${hPct}%)${extras}`,
        margin + 20,
        ry
      );
      ry += 16;
    });

    pdf.setFontSize(8);
    pdf.setTextColor(150, 150, 170);
    const ts = item.scannedAt
      ? `Scanned ${new Date(item.scannedAt).toLocaleString()}`
      : '';
    pdf.text(ts, margin, pageH - 16);
    pdf.text(
      `Page ${i + 2} of ${done.length + 1}`,
      pageW - margin,
      pageH - 16,
      { align: 'right' }
    );
  }

  onProgress?.('Saving PDF…', 0.98);
  pdf.save(filename);
  onProgress?.('Done', 1);
}

export async function downloadAnnotatedImage(item: PredictImageState): Promise<void> {
  const annotated = await buildAnnotatedImage(item);
  if (!annotated) throw new Error('No prediction to annotate');
  const a = document.createElement('a');
  a.href = annotated.dataUrl;
  a.download = (item.sourceName.replace(/\.[^/.]+$/, '') || 'annotated') + '_annotated.jpg';
  a.click();
}
