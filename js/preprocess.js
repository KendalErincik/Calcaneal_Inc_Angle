/* Pre/post-processing for the ONNX per-class masks, ported to match the Python EXACTLY.
 *
 * Python pipeline (perclass_onnx.py):
 *   small = np.asarray(Image.fromarray(gray8).resize((W, H), Image.BILINEAR), float32) / 255
 *   logits = session.run(x=small[None,None])                 # NCHW
 *   prob   = 1 / (1 + exp(-logits))                          # sigmoid at (W,H)
 *   full   = cv2.resize(prob, (w0, h0), INTER_LINEAR)        # back to source res
 *   mask   = (full > threshold) * 255                        # uint8 0/255
 *
 * Parity notes:
 *  - The downscale is Pillow's BILINEAR resampler. Pillow uses an antialiasing
 *    (support-scaled) convolution in 8-bit FIXED POINT (PRECISION_BITS = 22), so we
 *    reproduce its integer coefficient path bit-for-bit -> identical uint8 output.
 *  - The upscale is OpenCV INTER_LINEAR on a float32 map = half-pixel-centre bilinear
 *    with border replication -> reproduced in plain float.
 *
 * Pure JS, no dependencies; usable in Node (module.exports) and the browser (window).
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory();
  else root.Preprocess = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var PRECISION_BITS = 32 - 8 - 2; // = 22, same as Pillow's Resample.c

  function clip8(v) {
    // Pillow clip8: >= (1<<PRECISION_BITS<<8) -> 255, <=0 -> 0, else v>>PRECISION_BITS
    if (v >= (1 << (PRECISION_BITS + 8))) return 255;
    if (v <= 0) return 0;
    return (v >> PRECISION_BITS) & 0xff;
  }

  // Pillow precompute_coeffs for a BILINEAR filter (support = 1.0) over a full resize
  // (in0 = 0, in1 = inSize). Returns integer coeffs + per-output bounds, matching the
  // 8bpc normalize path (coeffs scaled by 1<<PRECISION_BITS, rounded like Pillow).
  function precomputeCoeffs(inSize, outSize) {
    var scale = inSize / outSize;
    var filterscale = scale < 1.0 ? 1.0 : scale;
    var support = 1.0 * filterscale; // bilinear support = 1.0
    var ksize = (Math.ceil(support) | 0) * 2 + 1;

    var kk = new Int32Array(outSize * ksize);
    var bounds = new Int32Array(outSize * 2);
    var tmp = new Float64Array(ksize);

    for (var xx = 0; xx < outSize; xx++) {
      var center = (xx + 0.5) * scale; // in0 = 0
      var ss = 1.0 / filterscale;
      // Pillow: (int)(center - support + 0.5) truncates toward zero (center>=0)
      var xmin = (center - support + 0.5) | 0;
      if (xmin < 0) xmin = 0;
      var xmax = (center + support + 0.5) | 0;
      if (xmax > inSize) xmax = inSize;
      xmax -= xmin;

      var ww = 0.0;
      var x;
      for (x = 0; x < xmax; x++) {
        // bilinear_filter((x + xmin - center + 0.5) * ss)
        var t = (x + xmin - center + 0.5) * ss;
        if (t < 0) t = -t;
        var w = t < 1.0 ? 1.0 - t : 0.0;
        tmp[x] = w;
        ww += w;
      }
      for (x = 0; x < xmax; x++) {
        var kv = ww !== 0.0 ? tmp[x] / ww : 0.0;
        // normalize_coeffs_8bpc: round to integer with Pillow's +/-0.5 rule
        kk[xx * ksize + x] = kv < 0 ? ((-0.5 + kv * (1 << PRECISION_BITS)) | 0)
                                    : (( 0.5 + kv * (1 << PRECISION_BITS)) | 0);
      }
      for (; x < ksize; x++) kk[xx * ksize + x] = 0;
      bounds[xx * 2] = xmin;
      bounds[xx * 2 + 1] = xmax;
    }
    return { kk: kk, bounds: bounds, ksize: ksize };
  }

  // One horizontal 8bpc pass: src (srcW x H) uint8 -> dst (outW x H) uint8.
  function resampleH8(src, srcW, H, outW, co) {
    var out = new Uint8Array(outW * H);
    var kk = co.kk, bounds = co.bounds, ksize = co.ksize;
    var half = 1 << (PRECISION_BITS - 1);
    for (var yy = 0; yy < H; yy++) {
      var rowOff = yy * srcW;
      for (var xx = 0; xx < outW; xx++) {
        var xmin = bounds[xx * 2];
        var xmax = bounds[xx * 2 + 1];
        var koff = xx * ksize;
        var ss0 = half;
        for (var x = 0; x < xmax; x++) ss0 += src[rowOff + x + xmin] * kk[koff + x];
        out[yy * outW + xx] = clip8(ss0);
      }
    }
    return out;
  }

  // One vertical 8bpc pass: src (W x srcH) uint8 -> dst (W x outH) uint8.
  function resampleV8(src, W, srcH, outH, co) {
    var out = new Uint8Array(W * outH);
    var kk = co.kk, bounds = co.bounds, ksize = co.ksize;
    var half = 1 << (PRECISION_BITS - 1);
    for (var yy = 0; yy < outH; yy++) {
      var ymin = bounds[yy * 2];
      var ymax = bounds[yy * 2 + 1];
      var koff = yy * ksize;
      for (var xx = 0; xx < W; xx++) {
        var ss0 = half;
        for (var y = 0; y < ymax; y++) ss0 += src[(y + ymin) * W + xx] * kk[koff + y];
        out[yy * W + xx] = clip8(ss0);
      }
    }
    return out;
  }

  // Pillow BILINEAR resize of an 8-bit grayscale image (srcW x srcH) -> (dstW x dstH).
  // Horizontal pass then vertical pass, uint8 intermediate, exactly like ImagingResample.
  function pilResizeBilinear8(src, srcW, srcH, dstW, dstH) {
    var img = src, w = srcW, h = srcH;
    if (dstW !== w) {
      var coH = precomputeCoeffs(w, dstW);
      img = resampleH8(img, w, h, dstW, coH);
      w = dstW;
    }
    if (dstH !== h) {
      var coV = precomputeCoeffs(h, dstH);
      img = resampleV8(img, w, h, dstH, coV);
      h = dstH;
    }
    return img; // Uint8Array(dstW * dstH)
  }

  // Model input: PIL-resize gray8 to (W,H), /255 -> Float32Array NCHW (1,1,H,W).
  function preprocessInput(gray8, srcW, srcH, W, H) {
    var small = pilResizeBilinear8(gray8, srcW, srcH, W, H);
    var f = new Float32Array(W * H);
    for (var i = 0; i < f.length; i++) f[i] = small[i] / 255.0;
    return f; // row-major H*W, i.e. NCHW with N=C=1
  }

  // OpenCV INTER_LINEAR resize of a float32 map (srcW x srcH) -> (dstW x dstH).
  // Half-pixel centres, border replication; matches cv2.resize on CV_32F.
  function cv2ResizeLinearF32(src, srcW, srcH, dstW, dstH) {
    var out = new Float32Array(dstW * dstH);
    var sx = srcW / dstW, sy = srcH / dstH;
    for (var dy = 0; dy < dstH; dy++) {
      var fy = (dy + 0.5) * sy - 0.5;
      var y0 = Math.floor(fy);
      var wy = fy - y0;
      if (y0 < 0) { y0 = 0; wy = 0; }
      if (y0 >= srcH - 1) { y0 = srcH - 1 > 0 ? srcH - 2 : 0; wy = srcH - 1 > 0 ? 1 : 0; }
      var y1 = y0 + 1 < srcH ? y0 + 1 : y0;
      for (var dx = 0; dx < dstW; dx++) {
        var fx = (dx + 0.5) * sx - 0.5;
        var x0 = Math.floor(fx);
        var wx = fx - x0;
        if (x0 < 0) { x0 = 0; wx = 0; }
        if (x0 >= srcW - 1) { x0 = srcW - 1 > 0 ? srcW - 2 : 0; wx = srcW - 1 > 0 ? 1 : 0; }
        var x1 = x0 + 1 < srcW ? x0 + 1 : x0;
        var v00 = src[y0 * srcW + x0], v01 = src[y0 * srcW + x1];
        var v10 = src[y1 * srcW + x0], v11 = src[y1 * srcW + x1];
        var top = v00 + (v01 - v00) * wx;
        var bot = v10 + (v11 - v10) * wx;
        out[dy * dstW + dx] = top + (bot - top) * wy;
      }
    }
    return out;
  }

  function sigmoidInplace(a) {
    for (var i = 0; i < a.length; i++) a[i] = 1.0 / (1.0 + Math.exp(-a[i]));
    return a;
  }

  // logits (W x H, model output) -> uint8 0/255 mask at (srcW x srcH).
  // sigmoid at (W,H) -> cv2 upscale to source -> threshold.  Matches Python order.
  function maskFromLogits(logits, W, H, srcW, srcH, threshold) {
    var prob = new Float32Array(W * H);
    for (var i = 0; i < prob.length; i++) prob[i] = 1.0 / (1.0 + Math.exp(-logits[i]));
    var full = cv2ResizeLinearF32(prob, W, H, srcW, srcH);
    var out = new Uint8Array(srcW * srcH);
    var thr = threshold;
    for (var j = 0; j < full.length; j++) out[j] = full[j] > thr ? 255 : 0;
    return out; // { use as {w:srcW, h:srcH, data: out} for the geometry }
  }

  return {
    PRECISION_BITS: PRECISION_BITS,
    pilResizeBilinear8: pilResizeBilinear8,
    preprocessInput: preprocessInput,
    cv2ResizeLinearF32: cv2ResizeLinearF32,
    sigmoidInplace: sigmoidInplace,
    maskFromLogits: maskFromLogits,
  };
});
