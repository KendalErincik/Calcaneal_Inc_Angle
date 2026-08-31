/* Browser CIA demo - glue + overlay.
 * file -> DICOM/PNG decode (dicom.js, JPEG 2000 via openjpeg) -> per-class masks (ort_infer.js,
 * onnxruntime-web) -> deterministic geometry (foot_geom.js) -> canvas overlay.
 * Everything client-side. No uploads.
 */
(function () {
  "use strict";
  var $ = function (s) { return document.querySelector(s); };

  var j2kDecode = null;     // set once openjpeg is ready
  var modelsReady = false;

  // ---------- boot: wire onnxruntime-web + openjpeg + load the model manifest ----------
  function boot() {
    try {
      // absolute URL so onnxruntime-web's dynamic import() of the .mjs glue resolves
      ort.env.wasm.wasmPaths = new URL("vendor/ort/", document.baseURI).href;
      // multi-thread WASM when the page is cross-origin-isolated (COOP/COEP);
      // otherwise fall back to single-thread so a plain static host still works.
      var iso = (typeof self !== "undefined" && self.crossOriginIsolated);
      ort.env.wasm.numThreads = (iso && navigator.hardwareConcurrency)
        ? Math.min(4, navigator.hardwareConcurrency) : 1;
      ort.env.wasm.simd = true;
    } catch (e) { /* ort may set later */ }
    OrtInfer.configure({ ort: window.ort, modelsBaseUrl: "models/" });

    // openjpeg (JPEG 2000). Prefer the fast WASM decode build; fall back to the JS build.
    var ojFactory = window.OpenJPEGWASM || window.OpenJPEGJS;
    if (typeof ojFactory === "function") {
      var ojArg = { locateFile: function (p) { return new URL("vendor/openjpeg/" + p, document.baseURI).href; } };
      ojFactory(ojArg).then(function (oj) { j2kDecode = DicomDecode.makeOpenJpegDecoder(oj); })
        .catch(function () { /* leave null; uncompressed still works */ });
    }

    OrtInfer.loadManifest().then(function () {
      modelsReady = true;
      $("#boot").style.display = "none";
      $("#foot-pick").disabled = false;
    }).catch(function (e) {
      $("#boot").className = "status warnbox";
      $("#boot").innerHTML = "Models could not be loaded from <code>models/</code>. " +
        "Make sure this page is served over http (see README) and reload. (" + e.message + ")";
    });
  }

  function setStatus(id, msg, cls) {
    var el = $(id);
    el.style.display = "block";
    el.className = "status" + (cls ? " " + cls : "");
    el.innerHTML = msg;
  }

  function wireDrop(dropSel, inputSel, onFile) {
    var drop = $(dropSel), input = $(inputSel);
    input.onchange = function () { if (input.files[0]) onFile(input.files[0]); };
    ["dragenter", "dragover"].forEach(function (e) {
      drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.add("over"); });
    });
    ["dragleave", "drop"].forEach(function (e) {
      drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.remove("over"); });
    });
    drop.addEventListener("drop", function (ev) { var f = ev.dataTransfer.files[0]; if (f) onFile(f); });
  }

  // ---------- input decode (DICOM or PNG/JPG) ----------
  var IMG_EXT = /\.(png|jpe?g|bmp|gif|tiff?|webp)$/i;
  var DICOM_EXT = /\.(dcm|dicom|ima)$/i;

  function imageToGray(buffer, mime) {
    // PNG/JPG -> uint8 grayscale (ITU-R 601 luma, like PIL 'L').
    return createImageBitmap(new Blob([buffer], { type: mime || "image/png" })).then(function (bmp) {
      var c = document.createElement("canvas"); c.width = bmp.width; c.height = bmp.height;
      var ctx = c.getContext("2d"); ctx.drawImage(bmp, 0, 0);
      var d = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      var g = new Uint8Array(bmp.width * bmp.height);
      for (var i = 0; i < g.length; i++) {
        g[i] = (d[i * 4] * 299 + d[i * 4 + 1] * 587 + d[i * 4 + 2] * 114) / 1000 | 0;
      }
      return { gray: g, w: bmp.width, h: bmp.height };
    });
  }

  function decodeInput(file, fast) {
    return file.arrayBuffer().then(function (buffer) {
      var name = file.name || "";
      var looksImage = IMG_EXT.test(name) && !DICOM_EXT.test(name);
      var opts = { dicomParser: window.dicomParser, decode: j2kDecode,
                   fast: !!fast, maxLongSide: 2048 };

      function asDicom() {
        var ds = DicomDecode.parse(buffer, window.dicomParser);
        return DicomDecode.decodeFoot(ds, opts);
      }
      if (looksImage) {
        return imageToGray(buffer, file.type).catch(function () { return asDicom(); });
      }
      try { return Promise.resolve(asDicom()); }
      catch (e) { return imageToGray(buffer, file.type); }
    });
  }

  // ---------- overlay drawing ----------
  function grayToCanvas(canvas, gray, w, h) {
    canvas.width = w; canvas.height = h;
    var ctx = canvas.getContext("2d");
    var img = ctx.createImageData(w, h), d = img.data;
    for (var i = 0; i < w * h; i++) { var v = gray[i]; d[i * 4] = v; d[i * 4 + 1] = v; d[i * 4 + 2] = v; d[i * 4 + 3] = 255; }
    ctx.putImageData(img, 0, 0);
    return ctx;
  }
  function rgb(a) { return "rgb(" + a[0] + "," + a[1] + "," + a[2] + ")"; }
  function line(ctx, a, b, color, lw) {
    if (!a || !b) return;
    ctx.strokeStyle = rgb(color); ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  function ring(ctx, p, color, r, lw) {
    if (!p) return;
    ctx.strokeStyle = rgb(color); ctx.lineWidth = lw;
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI); ctx.stroke();
  }

  var GROUND = [24, 210, 255], INCLIN = [255, 59, 48];
  function drawFoot(canvas, gray, w, h, geo) {
    var ctx = grayToCanvas(canvas, gray, w, h);
    var r = Math.max(4, w / 200 | 0), lw = Math.max(2, w / 350 | 0);
    line(ctx, geo.P1, geo.P2, GROUND, lw);
    line(ctx, geo.P4, geo.P3, INCLIN, lw);
    ring(ctx, geo.P1, GROUND, r, Math.max(2, r / 2 | 0));
    ring(ctx, geo.P2, GROUND, r, Math.max(2, r / 2 | 0));
    ring(ctx, geo.P3, INCLIN, r, Math.max(2, r / 2 | 0));
    ring(ctx, geo.P4, INCLIN, r, Math.max(2, r / 2 | 0));
    if (geo.cia != null) {
      ctx.fillStyle = "rgb(255,214,10)";
      ctx.font = Math.max(14, w / 55 | 0) + "px sans-serif"; ctx.textBaseline = "top";
      ctx.fillText("CIA = " + geo.cia.toFixed(1) + " deg", 10, 10);
    }
  }

  function showCanvas(canvasSel, saveSel) {
    var c = $(canvasSel); c.style.display = "block";
    var save = $(saveSel);
    try { save.href = c.toDataURL("image/png"); save.style.display = "inline-block"; } catch (e) { /* ignore */ }
  }

  // ---------- FOOT ----------
  function runFoot(file) {
    if (!modelsReady) return;
    $("#foot-canvas").style.display = "none"; $("#foot-save").style.display = "none";
    $("#foot-status").style.display = "none"; $("#foot-spin").style.display = "block";
    var tf0 = performance.now();
    decodeInput(file, $("#foot-fast") && $("#foot-fast").checked).then(function (dec) {
      var tf1 = performance.now();
      return OrtInfer.runGroup("foot", dec.gray, dec.w, dec.h).then(function (masks) {
        console.log("foot ms | decode=" + ((tf1 - tf0) | 0) + " infer(3 models)=" + ((performance.now() - tf1) | 0) +
          " | " + dec.w + "x" + dec.h);
        var geo = FootGeom.geometryFromMasks(masks, null);
        drawFoot($("#foot-canvas"), dec.gray, dec.w, dec.h, geo);
        $("#foot-spin").style.display = "none";
        if (geo.cia == null) {
          var miss = [];
          if (!geo.P2) miss.push("calcaneus"); if (!geo.P1) miss.push("metatarsal_5");
          if (!geo.P4) miss.push("calcaneonavicular"); if (!geo.P3) miss.push("tangent");
          setStatus("#foot-status", "Angle not computed" + (miss.length ? " (missing: " + miss.join(", ") + ")" : "") +
            ". Try a clearer lateral foot DICOM.", "err");
        } else {
          setStatus("#foot-status", '<div class="big">CIA = ' + geo.cia.toFixed(1) + "&deg;</div>", "ok");
        }
        showCanvas("#foot-canvas", "#foot-save");
      });
    }).catch(function (e) {
      $("#foot-spin").style.display = "none";
      setStatus("#foot-status", "Could not measure: " + (e && e.message ? e.message : e), "err");
    });
  }

  // ---------- init ----------
  document.addEventListener("DOMContentLoaded", function () {
    $("#foot-pick").onclick = function () { $("#foot-file").click(); };
    wireDrop("#foot-drop", "#foot-file", runFoot);
    boot();
  });
})();
