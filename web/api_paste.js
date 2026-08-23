// Paste an API-format workflow onto the canvas. The stock paste handler only
// understands UI-format JSON (nodes/links); API JSON (a dict of class_type
// nodes, what /prompt takes and what the deck's "api" links hand out) is
// silently ignored. This catches it first and routes it through the frontend's
// own loadApiJson, which builds widgets from the server schema (no positional
// guessing). loadApiJson replaces the current graph, so a non-empty canvas asks.
import { app } from "../../scripts/app.js";

function isApiJson(d) {
  if (typeof app.isApiJson === "function") return app.isApiJson(d);
  return d && typeof d === "object" && !Array.isArray(d) && !("nodes" in d) &&
    Object.values(d).length > 0 && Object.values(d).every((n) => n && typeof n === "object" && n.class_type);
}

app.registerExtension({
  name: "MAINodes.ApiPaste",
  setup() {
    window.addEventListener("paste", async (e) => {
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const text = e.clipboardData?.getData("text/plain");
      if (!text || text.length < 20 || text.trim()[0] !== "{") return;
      let data;
      try { data = JSON.parse(text); } catch { return; }
      if (!isApiJson(data)) return;                      // UI JSON and everything else: stock path
      e.preventDefault(); e.stopImmediatePropagation();
      const n = app.graph?._nodes?.length || 0;
      if (n && !confirm(`Paste an API-format workflow (${Object.keys(data).length} nodes)? It replaces the ${n} nodes on this canvas.`)) return;
      try { await app.loadApiJson(data, "pasted_api.json"); }
      catch (err) { console.error("MAINodes api paste failed:", err); alert("API workflow paste failed: " + err); }
    }, true);
  },
});
