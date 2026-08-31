/* In-browser DICOM intake, ported to match the desktop Python EXACTLY.
 *
 * Two decode paths (they differ - keep them separate, like the Python):
 *   decodeFoot(ds)  == foot_geometry.dicom_normalize
 *       pixel_array -> (RGB mean) -> MONOCHROME1 invert -> WindowCenter/Width CLIP
 *       -> min-max 0..255 uint8 -> mask top 8% / bottom 6% (collimator).
 *   downscale / resolveMm / magNote == leg_geometry (downscale reuses the PIL-exact
 *   bilinear from preprocess.js so the working image matches the desktop bit-for-bit).
 *
 * Parsing uses dicom-parser (uncompressed transfer syntaxes; compressed = later phase,
 * handled by a clear error here). All normalisation is done in this file, which is where
 * manuscript parity matters. Verified in Node against pydicom on synthetic DICOMs.
 *
 * Pure JS; Node (module.exports, requires ./preprocess.js + dicom-parser) and browser
 * (window.DicomDecode; expects window.dicomParser + window.Preprocess).
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory(require("./preprocess.js"));
  else root.DicomDecode = factory(root.Preprocess);
})(typeof self !== "undefined" ? self : this, function (Preprocess) {
  "use strict";

  // transfer syntaxes we can read raw (little/big endian, uncompressed)
  var UNCOMPRESSED = {
    "1.2.840.10008.1.2": true,     // Implicit VR LE
    "1.2.840.10008.1.2.1": true,   // Explicit VR LE
    "1.2.840.10008.1.2.2": true,   // Explicit VR BE (retired)
  };
  var BIG_ENDIAN = { "1.2.840.10008.1.2.2": true };
  // encapsulated JPEG 2000 (our data). Lossless (.90) is bit-exact; .91 is lossy but the
  // same decoder handles both. Other compressions (JPEG/JPEG-LS/RLE) can be added the same way.
  var J2K = {
    "1.2.840.10008.1.2.4.90": true, // JPEG 2000 Lossless Only
    "1.2.840.10008.1.2.4.91": true, // JPEG 2000
  };

  function _has(ds, tag) {
    return !!(ds.elements && ds.elements[tag]);
  }
  function _floats(ds, tag) {
    // all backslash-separated values of a DS/IS element as numbers
    if (!_has(ds, tag)) return [];
    var s = ds.string(tag);
    if (s == null || s === "") return [];
    return String(s).split("\\").map(function (v) { return parseFloat(v); })
      .filter(function (v) { return !isNaN(v); });
  }
  function _float0(ds, tag) {
    var a = _floats(ds, tag);
    return a.length ? a[0] : null;
  }

  // ---- read the raw stored pixel values (respecting BitsAllocated + sign) ----
  function _meta(ds) {
    var tsn = ds.string("x00020010") || "1.2.840.10008.1.2.1";
    var m = {
      tsn: tsn,
      rows: ds.uint16("x00280010"),
      cols: ds.uint16("x00280011"),
      samples: ds.uint16("x00280002") || 1,
      photometric: (ds.string("x00280004") || "").toUpperCase(),
      bits: ds.uint16("x00280100") || 16,
      bitsStored: ds.uint16("x00280101") || (ds.uint16("x00280100") || 16),
      signed: (ds.uint16("x00280103") || 0) === 1,
      planar: ds.uint16("x00280006") || 0,
      little: !BIG_ENDIAN[tsn],
      voifunc: (ds.string("x00281056") || "LINEAR").toUpperCase(),
      slope: _has(ds, "x00281053") ? _float0(ds, "x00281053") : null,
      intercept: _has(ds, "x00281052") ? _float0(ds, "x00281052") : null,
      wc: _has(ds, "x00281050") ? _floats(ds, "x00281050") : null,
      ww: _has(ds, "x00281051") ? _floats(ds, "x00281051") : null,
    };
    m.hasWindow = m.wc && m.ww && m.wc.length > 0 && m.ww.length > 0;
    return m;
  }

  function _pixelView(ds) {
    var el = ds.elements["x7fe00010"];
    if (!el) throw new Error("DICOM has no pixel data.");
    if (el.encapsulatedPixelData) {
      throw new Error("Encapsulated/compressed pixel data not supported yet.");
    }
    var ba = ds.byteArray;
    return new DataView(ba.buffer, ba.byteOffset + el.dataOffset, el.length);
  }

  // Raw stored values as Float64Array, length rows*cols*samples (interleaved if RGB).
  function _readStored(ds, m) {
    var dv = _pixelView(ds);
    var n = m.rows * m.cols * m.samples;
    var out = new Float64Array(n);
    var le = m.little;
    if (m.bits === 8) {
      for (var i = 0; i < n; i++) out[i] = m.signed ? dv.getInt8(i) : dv.getUint8(i);
    } else if (m.bits === 16) {
      for (var j = 0; j < n; j++) out[j] = m.signed ? dv.getInt16(j * 2, le) : dv.getUint16(j * 2, le);
    } else {
      throw new Error("Unsupported BitsAllocated: " + m.bits);
    }
    return out;
  }

  // Assemble decoded (uncompressed) codec bytes into stored sample values (Float64Array,
  // length rows*cols*samples). Uses the codec's frameInfo for element size + sign.
  function _assembleDecoded(decoded, fi) {
    var comps = fi.componentCount || 1;
    var n = fi.width * fi.height * comps;
    var out = new Float64Array(n);
    var elem = (fi.bitsPerSample > 8) ? 2 : 1;
    // openjpeg outputs native little-endian; use a typed-array view (fast) rather than a
    // per-pixel DataView loop - matters a lot on 20+ MP long-leg films.
    var i, src;
    if (elem === 1) {
      src = fi.isSigned ? new Int8Array(decoded.buffer, decoded.byteOffset, n)
                        : new Uint8Array(decoded.buffer, decoded.byteOffset, n);
    } else {
      src = fi.isSigned ? new Int16Array(decoded.buffer, decoded.byteOffset, n)
                        : new Uint16Array(decoded.buffer, decoded.byteOffset, n);
    }
    for (i = 0; i < n; i++) out[i] = src[i];
    return out;
  }

  // Stored sample values for a frame -> { data:Float64Array, w, h, comps }. Handles
  // uncompressed AND encapsulated JPEG 2000. Async (WASM codecs are async). opts:
  //   { dicomParser, decode(frameBytes, info)->Promise<{decoded,frameInfo}>, frame,
  //     fast, maxLongSide }.  When fast + maxLongSide, JPEG 2000 is decoded at a reduced
  //   resolution level (decodeSubResolution) so a huge film decodes far faster; the result
  //   is APPROXIMATE (not bit-identical to a full decode) but geometry stays close.
  function _samples(ds, m, opts) {
    opts = opts || {};
    if (UNCOMPRESSED[m.tsn]) {
      return Promise.resolve({ data: _readStored(ds, m), w: m.cols, h: m.rows, comps: m.samples });
    }
    if (J2K[m.tsn]) {
      var dp = opts.dicomParser || (typeof self !== "undefined" ? self.dicomParser : null) || (typeof window !== "undefined" ? window.dicomParser : null);
      if (!dp) return Promise.reject(new Error("dicom-parser needed to extract the compressed frame."));
      if (typeof opts.decode !== "function") {
        return Promise.reject(new Error("Compressed DICOM (" + m.tsn + ") needs a decoder - pass opts.decode (e.g. cornerstone openjpeg)."));
      }
      var full = Math.max(m.rows, m.cols), level = 0;
      if (opts.fast && opts.maxLongSide && full > opts.maxLongSide) {
        level = Math.max(0, Math.floor(Math.log2(full / opts.maxLongSide)));
      }
      var pde = ds.elements["x7fe00010"];
      var frame = dp.readEncapsulatedImageFrame(ds, pde, opts.frame || 0);
      var info = { level: level, fullW: m.cols, fullH: m.rows, tsn: m.tsn };
      return Promise.resolve(opts.decode(frame, info)).then(function (r) {
        var fi = r.frameInfo;
        return { data: _assembleDecoded(r.decoded, fi), w: fi.width, h: fi.height, comps: fi.componentCount || 1 };
      });
    }
    return Promise.reject(new Error("Unsupported transfer syntax " + m.tsn + " (no decoder). Uncompressed + JPEG 2000 supported."));
  }

  // Collapse RGB -> single channel. mode "mean3" (foot) or "chan0" (leg).
  function _toMono(arr, comps, mode) {
    if (comps === 1) return arr;
    var np = (arr.length / comps) | 0, out = new Float64Array(np);
    for (var i = 0; i < np; i++) {
      if (mode === "mean3") out[i] = (arr[i * comps] + arr[i * comps + 1] + arr[i * comps + 2]) / 3.0;
      else out[i] = arr[i * comps]; // channel 0
    }
    return out;
  }

  // ---- pydicom apply_windowing (LINEAR default / LINEAR_EXACT / SIGMOID) ----
  // Throws (like pydicom) for non-monochrome or invalid width so the leg path can fall
  // back to raw, matching leg_geometry._read_pixel_array's try/except.
  function _applyWindowing(arr, m) {
    if (m.photometric !== "MONOCHROME1" && m.photometric !== "MONOCHROME2") {
      throw new Error("windowing needs MONOCHROME1/2");
    }
    var center = m.wc[0], width = m.ww[0];
    var yMin, yMax;
    if (m.signed) { yMin = -(Math.pow(2, m.bitsStored - 1)); yMax = Math.pow(2, m.bitsStored - 1) - 1; }
    else { yMin = 0; yMax = Math.pow(2, m.bitsStored) - 1; }
    if (m.slope != null && m.intercept != null) {
      yMin = yMin * m.slope + m.intercept;
      yMax = yMax * m.slope + m.intercept;
    }
    var yRange = yMax - yMin;
    var out = new Float64Array(arr.length);
    var vf = m.voifunc;
    if (vf === "LINEAR" || vf === "LINEAR_EXACT") {
      if (vf === "LINEAR") {
        if (width < 1) throw new Error("Window Width must be >= 1 for LINEAR");
        center -= 0.5; width -= 1;
      } else if (width <= 0) {
        throw new Error("Window Width must be > 0 for LINEAR_EXACT");
      }
      var lo = center - width / 2, hi = center + width / 2;
      for (var i = 0; i < arr.length; i++) {
        var v = arr[i];
        if (v <= lo) out[i] = yMin;
        else if (v > hi) out[i] = yMax;
        else out[i] = ((v - center) / width + 0.5) * yRange + yMin;
      }
    } else if (vf === "SIGMOID") {
      if (width <= 0) throw new Error("Window Width must be > 0 for SIGMOID");
      for (var k = 0; k < arr.length; k++) {
        out[k] = yRange / (1 + Math.exp(-4 * (arr[k] - center) / width)) + yMin;
      }
    } else {
      throw new Error("Unsupported VOI LUT Function: " + vf);
    }
    return out;
  }

  // ---- pydicom apply_voi (VOILUTSequence lookup table) ----
  function _voiLutSeq(ds) {
    var el = ds.elements["x00283010"];
    if (!el || !el.items || !el.items.length) return null;
    var item = el.items[0].dataSet;
    if (!item || !_has(item, "x00283002") || !_has(item, "x00283006")) return null;
    // LUTDescriptor: [nr_entries (US, 0->65536), first_map (signed possible), nbits]
    var nr = item.uint16("x00283002", 0) || 65536;
    var first = item.int16("x00283002", 1);
    var nbits = item.uint16("x00283002", 2);
    var de = item.elements["x00283006"];
    var ba = item.byteArray;
    var data;
    if (nbits === 8) {
      data = new Uint8Array(ba.buffer, ba.byteOffset + de.dataOffset, de.length);
    } else { // 10..16 -> uint16
      var count = de.length / 2, dv = new DataView(ba.buffer, ba.byteOffset + de.dataOffset, de.length);
      data = new Uint16Array(count);
      for (var i = 0; i < count; i++) data[i] = dv.getUint16(i * 2, true);
    }
    return { nr: nr, first: first, data: data };
  }

  function _applyVoi(arr, lut) {
    var out = new Float64Array(arr.length);
    var nr = lut.nr, first = lut.first, d = lut.data, last = nr - 1;
    for (var i = 0; i < arr.length; i++) {
      var iv = arr[i] >= first ? arr[i] - first : 0;
      if (iv < 0) iv = 0; else if (iv > last) iv = last;
      out[i] = d[iv];
    }
    return out;
  }

  // ---- leg _normalize_to_uint8: arr.astype(float32) -> min-max 0..255 -> MONO1 invert ----
  // Python casts the windowed array to float32 first, then normalises; reproduced with a
  // Float32Array (rounds on store) + Math.fround so the uint8 truncation matches bit-for-bit.
  var fr = Math.fround;
  function _normalizeLeg(arr, photometric) {
    var f = new Float32Array(arr.length);       // arr.astype(np.float32)
    for (var i = 0; i < arr.length; i++) f[i] = arr[i];
    var lo = Infinity, hi = -Infinity;
    for (var a = 0; a < f.length; a++) { var v = f[a]; if (v < lo) lo = v; if (v > hi) hi = v; }
    var out = new Uint8Array(f.length);
    if (hi > lo) {
      var den = hi - lo;                         // float64 python scalar
      for (var j = 0; j < f.length; j++) out[j] = Math.trunc(fr(fr(fr(f[j] - lo) / den) * 255.0));
    }
    if (photometric === "MONOCHROME1") for (var k = 0; k < out.length; k++) out[k] = 255 - out[k];
    return out;
  }

  // ---- PixelSpacing / magnification (leg_geometry._pixel_spacing) ----
  function _pixelSpacing(ds) {
    var sp = null;
    var a = _floats(ds, "x00280030"); if (a.length) sp = a[0];
    if (sp == null) { var b = _floats(ds, "x00181164"); if (b.length) sp = b[0]; }
    var calType = ds.string("x00280a02");
    var cal = !!(calType && calType.length) && _has(ds, "x00280030");
    var mag = null;
    var sid = _float0(ds, "x00181110"), sod = _float0(ds, "x00181111");
    if (sid && sod && sod > 0) mag = Math.round((sid / sod) * 10000) / 10000;
    if (mag == null) {
      var em = _float0(ds, "x00181114");
      if (em) mag = Math.round(em * 10000) / 10000;
    }
    return { sp: sp, cal: cal, mag: mag };
  }

  // ===================== public decode paths =====================

  // Foot: foot_geometry.dicom_normalize -> {gray, w, h}. Done in float32 (Float32Array +
  // Math.fround) to match numpy's pixel_array.astype(np.float32) pipeline bit-for-bit.
  // Async: opts = { dicomParser, decode } for compressed (JPEG 2000) files.
  function decodeFoot(ds, opts) {
    var m = _meta(ds);
    return _samples(ds, m, opts).then(function (s) {
    var cols = s.w, h = s.h;
    var mono = (s.comps >= 3) ? _toMono(s.data, s.comps, "mean3") : s.data;
    var px = new Float32Array(mono.length);          // astype(float32)
    for (var i = 0; i < mono.length; i++) px[i] = mono[i];
    // MONOCHROME1 invert using the float32 max (BEFORE windowing), like the Python
    if (m.photometric === "MONOCHROME1") {
      var mx = -Infinity; for (var i2 = 0; i2 < px.length; i2++) if (px[i2] > mx) mx = px[i2];
      for (var j = 0; j < px.length; j++) px[j] = mx - px[j]; // store rounds to float32
    }
    // WindowCenter & WindowWidth present -> simple clip (first value, no LINEAR shift)
    if (m.hasWindow) {
      var lo = m.wc[0] - m.ww[0] / 2, hi = m.wc[0] + m.ww[0] / 2;
      for (var k = 0; k < px.length; k++) { var v = px[k]; px[k] = v < lo ? lo : (v > hi ? hi : v); }
    }
    // min-max 0..255 in float32
    var mn = Infinity, mxx = -Infinity;
    for (var a = 0; a < px.length; a++) { var w = px[a]; if (w < mn) mn = w; if (w > mxx) mxx = w; }
    var img = new Uint8Array(px.length);
    if (mxx > mn) {
      var den = fr(mxx - mn);                        // np.float32 scalar
      for (var b = 0; b < px.length; b++) img[b] = Math.trunc(fr(fr(fr(px[b] - mn) / den) * 255.0));
    }
    // collimator mask: top 8%, bottom 6%
    var top = Math.trunc(h * 0.08), bot = Math.trunc(h * 0.06);
    for (var y = 0; y < top; y++) for (var x = 0; x < cols; x++) img[y * cols + x] = 0;
    for (var y2 = h - bot; y2 < h; y2++) for (var x2 = 0; x2 < cols; x2++) img[y2 * cols + x2] = 0;
    return { gray: img, w: cols, h: h };
    });
  }


  // Convenience: build a J2K decode fn from a cornerstone openjpeg instance (browser + node).
  // decode fn(frameBytes, info) where info = { level, fullW, fullH }. level>0 uses
  // decodeSubResolution (reduced-resolution / fast). getFrameInfo() reports the FULL size
  // even for a sub-resolution decode, so the reduced dims are computed as ceil(full/2^level)
  // (and sanity-checked against the decoded buffer length).
  function makeOpenJpegDecoder(openjpeg) {
    return function (frameBytes, info) {
      info = info || {};
      var level = info.level | 0;
      var decoder = new openjpeg.J2KDecoder();
      var enc = decoder.getEncodedBuffer(frameBytes.length);
      enc.set(frameBytes);
      if (level > 0 && typeof decoder.decodeSubResolution === "function") decoder.decodeSubResolution(level);
      else { level = 0; decoder.decode(); }
      var fi = decoder.getFrameInfo();
      var dec = decoder.getDecodedBuffer();
      var out = new Uint8Array(dec.length);   // copy: the codec reuses its heap between frames
      out.set(dec);
      var w = fi.width, h = fi.height, comps = fi.componentCount || 1;
      if (level > 0) {
        var div = Math.pow(2, level);
        w = Math.ceil((info.fullW || fi.width) / div);
        h = Math.ceil((info.fullH || fi.height) / div);
        var elem = (fi.bitsPerSample > 8) ? 2 : 1;
        // if the ceil guess disagrees with the buffer, fall back to floor
        if (w * h * comps * elem !== out.length) {
          var wf = Math.floor((info.fullW || fi.width) / div), hf = Math.floor((info.fullH || fi.height) / div);
          if (wf * hf * comps * elem === out.length) { w = wf; h = hf; }
        }
      }
      return { decoded: out, frameInfo: { width: w, height: h, bitsPerSample: fi.bitsPerSample, isSigned: fi.isSigned, componentCount: comps } };
    };
  }

  // leg_geometry.downscale: cap long side (default 2200) with PIL bilinear; returns {gray,w,h,r}
  function downscale(gray, w, h, maxDim) {
    maxDim = maxDim || 2200;
    var long = Math.max(w, h);
    if (long <= maxDim) return { gray: gray, w: w, h: h, r: 1.0 };
    var r = maxDim / long;
    var nw = Math.max(1, Math.round(w * r)), nh = Math.max(1, Math.round(h * r));
    var out = Preprocess.pilResizeBilinear8(gray, w, h, nw, nh);
    return { gray: out, w: nw, h: nh, r: r };
  }

  // leg_geometry.resolve_mm
  function resolveMm(sp, vendorCal, mag, correct) {
    if (!sp) return { mmpp: null, calibrated: false, method: "none", mag: null };
    if (vendorCal) return { mmpp: sp, calibrated: true, method: "vendor", mag: mag };
    if (correct && mag) { var mm = Number(mag); if (mm > 0) return { mmpp: sp / mm, calibrated: true, method: "mag", mag: mm }; }
    return { mmpp: sp, calibrated: true, method: "detector", mag: mag };
  }

  function magNote(method, mag) {
    if (method === "mag") return "magnification corrected (x" + (Math.round(Number(mag) * 1000) / 1000) + ") - true size";
    if (method === "detector") return "detector plane - matches PACS";
    if (method === "vendor") return "device-calibrated";
    return "not calibrated (no PixelSpacing)";
  }

  function parse(arrayBufferOrU8, dicomParser) {
    var dp = dicomParser || (typeof self !== "undefined" ? self.dicomParser : null) || (typeof window !== "undefined" ? window.dicomParser : null);
    if (!dp) throw new Error("dicom-parser not found - pass it in or load it globally.");
    var u8 = arrayBufferOrU8 instanceof Uint8Array ? arrayBufferOrU8 : new Uint8Array(arrayBufferOrU8);
    return dp.parseDicom(u8);
  }

  return {
    parse: parse,
    decodeFoot: decodeFoot,
    downscale: downscale,
    resolveMm: resolveMm,
    magNote: magNote,
    makeOpenJpegDecoder: makeOpenJpegDecoder,
    UNCOMPRESSED: UNCOMPRESSED,
    J2K: J2K,
  };
});
