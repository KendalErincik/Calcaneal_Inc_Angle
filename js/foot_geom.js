/* Pure-JS port of desktop-measure-foot/foot_geometry.py (Calcaneal Inclination Angle).
 * No OpenCV, no dependencies. Reproduces the deployed Python landmarks/CIA so the browser
 * build matches the manuscript. Verified in Node against the Python (test/run_foot.js).
 *
 * Mask format: { w, h, data:Uint8Array(w*h) }  (foreground = value > 0).
 * Coordinates are [x, y] (x = column, y = row), same as the Python (contour x,y).
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory();
  else root.FootGeom = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---- largest 8-connected component pixels of a mask (foreground > 0) ----
  function largestComponent(mask) {
    const { w, h, data } = mask;
    const lab = new Int32Array(w * h).fill(-1);
    const stack = new Int32Array(w * h);
    let best = null, bestCount = -1;
    for (let i = 0; i < w * h; i++) {
      if (data[i] === 0 || lab[i] !== -1) continue;
      // BFS/DFS this component
      let sp = 0; stack[sp++] = i; lab[i] = i;
      const px = [];
      while (sp > 0) {
        const p = stack[--sp];
        const y = (p / w) | 0, x = p - y * w;
        px.push(p);
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy; if (ny < 0 || ny >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx; if (nx < 0 || nx >= w) continue;
            const q = ny * w + nx;
            if (data[q] !== 0 && lab[q] === -1) { lab[q] = i; stack[sp++] = q; }
          }
        }
      }
      if (px.length > bestCount) { bestCount = px.length; best = px; }
    }
    if (!best) return null;
    const n = best.length;
    const xs = new Int32Array(n), ys = new Int32Array(n);
    for (let k = 0; k < n; k++) { const p = best[k]; const y = (p / w) | 0; xs[k] = p - y * w; ys[k] = y; }
    return { xs, ys, n };
  }

  function centroidX(mask) {
    const { w, h, data } = mask;
    let sum = 0, cnt = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (data[y * w + x] !== 0) { sum += x; cnt++; }
    return cnt === 0 ? null : sum / cnt;
  }

  // inferior-most point (max y); tie -> first in contour order ~ we match Python by
  // taking, among max-y pixels, the one cv2's contour returns first. Empirically that is
  // the max-y pixel with the SMALLEST x on the bottom row for these convex masks.
  function argmaxY(comp) {
    let ymax = -1;
    for (let k = 0; k < comp.n; k++) if (comp.ys[k] > ymax) ymax = comp.ys[k];
    let xsel = Infinity;
    for (let k = 0; k < comp.n; k++) if (comp.ys[k] === ymax && comp.xs[k] < xsel) xsel = comp.xs[k];
    return [xsel, ymax];
  }

  // posterior-inferior corner: argmax(x+y) or argmax(y-x) if flipped.
  // Tie-break matches cv2's contour argmax first-occurrence: among equal score, the
  // pixel with the SMALLEST x wins (verified against the Python on tied cases, both
  // orientations).
  function argmaxScore(comp, flipped) {
    let best = -Infinity, bx = Infinity, by = 0;
    for (let k = 0; k < comp.n; k++) {
      const s = flipped ? (comp.ys[k] - comp.xs[k]) : (comp.xs[k] + comp.ys[k]);
      if (s > best || (s === best && comp.xs[k] < bx)) { best = s; bx = comp.xs[k]; by = comp.ys[k]; }
    }
    return [bx, by];
  }

  function detectFlipped(masks) {
    const fore = centroidX(masks.metatarsal_5);
    const heel = centroidX(masks.calcaneus_inferior);
    if (fore === null || heel === null) return null;
    return fore < heel;
  }

  // lower envelope of the calcaneus component: per x-bin inferior-most point, then smooth
  function extractInferiorContour(mask, smoothWindow, binSize) {
    smoothWindow = smoothWindow === undefined ? 8 : smoothWindow;
    binSize = binSize === undefined ? 3 : binSize;
    const comp = largestComponent(mask);
    if (!comp || comp.n < 5) return null;
    // bottom-most y per column x
    let xmin = Infinity, xmax = -Infinity;
    for (let k = 0; k < comp.n; k++) { if (comp.xs[k] < xmin) xmin = comp.xs[k]; if (comp.xs[k] > xmax) xmax = comp.xs[k]; }
    const colBottom = new Map(); // x -> max y
    for (let k = 0; k < comp.n; k++) {
      const x = comp.xs[k], y = comp.ys[k];
      const cur = colBottom.get(x);
      if (cur === undefined || y > cur) colBottom.set(x, y);
    }
    const inf = [];
    for (let b = xmin; b < xmax + binSize; b += binSize) {
      let byY = -1, bxX = null;
      for (let x = b; x < b + binSize; x++) {
        const y = colBottom.get(x);
        if (y === undefined) continue;
        if (y > byY) { byY = y; bxX = x; }
      }
      if (bxX !== null) inf.push([bxX, byY]);
    }
    if (inf.length < 3) return null;
    inf.sort((a, b) => a[0] - b[0]);
    if (smoothWindow > 1 && inf.length > smoothWindow) {
      const out = [];
      const W = smoothWindow;
      for (let i = 0; i + W <= inf.length; i++) {
        let sx = 0, sy = 0;
        for (let j = 0; j < W; j++) { sx += inf[i + j][0]; sy += inf[i + j][1]; }
        out.push([sx / W, sy / W]);
      }
      return out;
    }
    return inf;
  }

  // ---- convex hull (Andrew monotone chain), returns hull vertices CCW, no collinear ----
  function convexHull(points) {
    const pts = points.slice().sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    if (pts.length <= 2) return pts.slice();
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [];
    for (const p of pts) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) {
      const p = pts[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper);
  }

  function lowerTangent(P4, contour, minDist) {
    minDist = minDist === undefined ? 10.0 : minDist;
    const far = [];
    for (const p of contour) {
      const dx = p[0] - P4[0], dy = p[1] - P4[1];
      if (Math.sqrt(dx * dx + dy * dy) >= minDist) far.push(p);
    }
    if (far.length < 3) return null;
    const hull = convexHull(far);
    const n = hull.length;
    if (n < 3) return null;
    const cands = [];
    for (let i = 0; i < n; i++) {
      const V = hull[i], Vp = hull[(i - 1 + n) % n], Vn = hull[(i + 1) % n];
      const dx = V[0] - P4[0], dy = V[1] - P4[1];
      if (Math.sqrt(dx * dx + dy * dy) < 1e-6) continue;
      const sp = dx * (Vp[1] - V[1]) - dy * (Vp[0] - V[0]);
      const sn = dx * (Vn[1] - V[1]) - dy * (Vn[0] - V[0]);
      if (sp * sn >= -1e-6) cands.push(V);
    }
    if (cands.length === 0) return null;
    let best = cands[0];
    for (const c of cands) if (c[1] > best[1]) best = c;
    return best;
  }

  function computeCIA(P1, P2, P3, P4) {
    const vgx = P2[0] - P1[0], vgy = P2[1] - P1[1];
    const vtx = P3[0] - P4[0], vty = P3[1] - P4[1];
    const dot = vgx * vtx + vgy * vty;
    const ng = Math.hypot(vgx, vgy), nt = Math.hypot(vtx, vty);
    let cos = dot / (ng * nt + 1e-10);
    cos = Math.max(-1, Math.min(1, cos));
    const ang = Math.acos(cos) * 180 / Math.PI;
    return Math.min(ang, 180 - ang);
  }

  function geometryFromMasks(masks, flipped) {
    if (flipped === undefined || flipped === null) {
      flipped = detectFlipped(masks);
      if (flipped === null) flipped = false;
    }
    const cMt5 = largestComponent(masks.metatarsal_5);
    const cCal = largestComponent(masks.calcaneus_inferior);
    const cNav = largestComponent(masks.calcaneonavicular);
    const P1 = cMt5 ? argmaxY(cMt5) : null;
    const P2 = cCal ? argmaxY(cCal) : null;
    const P4 = cNav ? argmaxScore(cNav, flipped) : null;
    let P3 = null, cia = null;
    if (P1 && P2 && P4) {
      const contour = extractInferiorContour(masks.calcaneus_inferior);
      if (contour) {
        P3 = lowerTangent(P4, contour);
        if (P3) cia = computeCIA(P1, P2, P3, P4);
      }
    }
    return { cia, P1, P2, P3, P4, flipped };
  }

  return { geometryFromMasks, largestComponent, centroidX, convexHull, computeCIA,
           extractInferiorContour, lowerTangent, detectFlipped };
});
