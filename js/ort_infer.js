/* onnxruntime-web inference for the per-class U-Net masks (Foot: 3 models).
 *
 * Same pre/post-processing as the desktop Python (perclass_onnx.py),
 * implemented in preprocess.js and verified against PIL/cv2 in Node. This module only
 * orchestrates: load the single-file ONNX models (models_manifest.json), run each class,
 * and return masks ready for the pure-JS geometry (foot_geom.js).
 *
 * Usage (browser):
 *   OrtInfer.configure({ ort: window.ort, modelsBaseUrl: "models/", wasmPaths: "vendor/" });
 *   await OrtInfer.loadManifest();
 *   const masks = await OrtInfer.runGroup("foot", gray8, w, h);   // {label:{w,h,data}}
 *   const geo = FootGeom.geometryFromMasks(masks, null);
 *
 * `gray8` = Uint8Array(w*h), row-major, the SAME grayscale the Python normalises to
 * (produced by dicom.js in P3). Masks come back at the source (w,h).
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory(require("./preprocess.js"));
  else root.OrtInfer = factory(root.Preprocess);
})(typeof self !== "undefined" ? self : this, function (Preprocess) {
  "use strict";

  var _ort = null;                 // the onnxruntime-web namespace
  var _base = "models/";           // where the .onnx + manifest live
  var _manifest = null;            // { foot:{labels:{...}} }
  var _sessions = {};              // label -> { session, W, H, threshold, inputName }

  function configure(opts) {
    opts = opts || {};
    if (opts.ort) _ort = opts.ort;
    if (opts.modelsBaseUrl) _base = opts.modelsBaseUrl.replace(/\/?$/, "/");
    if (opts.wasmPaths && _ort && _ort.env && _ort.env.wasm) _ort.env.wasm.wasmPaths = opts.wasmPaths;
    return this;
  }

  function _ortNS() {
    var o = _ort || (typeof self !== "undefined" ? self.ort : null) || (typeof window !== "undefined" ? window.ort : null);
    if (!o) throw new Error("onnxruntime-web (ort) not found - load it and call OrtInfer.configure({ort}).");
    return o;
  }

  async function loadManifest() {
    var res = await fetch(_base + "models_manifest.json", { cache: "force-cache" });
    if (!res.ok) throw new Error("Cannot load models_manifest.json (" + res.status + ")");
    _manifest = await res.json();
    return _manifest;
  }

  function _labelMeta(label) {
    if (!_manifest) throw new Error("Call loadManifest() first.");
    for (var g in _manifest) {
      var labels = (_manifest[g] && _manifest[g].labels) || {};
      if (labels[label]) return labels[label];
    }
    throw new Error("Unknown label in manifest: " + label);
  }

  function groupLabels(group) {
    if (!_manifest) throw new Error("Call loadManifest() first.");
    var labels = (_manifest[group] && _manifest[group].labels) || {};
    return Object.keys(labels);
  }

  async function _session(label) {
    if (_sessions[label]) return _sessions[label];
    var meta = _labelMeta(label);
    var ort = _ortNS();
    var opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    var session = await ort.InferenceSession.create(_base + meta.onnx, opts);
    var entry = {
      session: session,
      W: meta.W | 0,
      H: meta.H | 0,
      threshold: typeof meta.threshold === "number" ? meta.threshold : 0.5,
      inputName: session.inputNames && session.inputNames[0] ? session.inputNames[0] : "input",
      outputName: session.outputNames && session.outputNames[0] ? session.outputNames[0] : "logits",
    };
    _sessions[label] = entry;
    return entry;
  }

  // Run one class -> uint8 mask {w,h,data} at the source resolution.
  async function runLabel(label, gray8, w, h) {
    var s = await _session(label);
    var ort = _ortNS();
    var input = Preprocess.preprocessInput(gray8, w, h, s.W, s.H); // Float32Array(H*W), /255
    var tensor = new ort.Tensor("float32", input, [1, 1, s.H, s.W]);
    var feeds = {};
    feeds[s.inputName] = tensor;
    var out = await s.session.run(feeds);
    var logits = out[s.outputName].data; // Float32Array(H*W)
    var data = Preprocess.maskFromLogits(logits, s.W, s.H, w, h, s.threshold);
    return { w: w, h: h, data: data };
  }

  // Run a set of labels -> { label: {w,h,data} }.
  async function runLabels(labels, gray8, w, h) {
    var out = {};
    for (var i = 0; i < labels.length; i++) out[labels[i]] = await runLabel(labels[i], gray8, w, h);
    return out;
  }

  // Run every class of the group ("foot").
  async function runGroup(group, gray8, w, h) {
    return runLabels(groupLabels(group), gray8, w, h);
  }

  // Warm the sessions (compile the graphs) without an image.
  async function warm(group) {
    var labels = groupLabels(group);
    for (var i = 0; i < labels.length; i++) await _session(labels[i]);
  }

  return {
    configure: configure,
    loadManifest: loadManifest,
    groupLabels: groupLabels,
    runLabel: runLabel,
    runLabels: runLabels,
    runGroup: runGroup,
    warm: warm,
  };
});
