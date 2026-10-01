"""Assemble the report page: template.html + report_data.json + report.js -> index.html (one self-contained file)."""
import json, os, re

H = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(H, "template.html"), encoding="utf-8").read()
data = json.load(open(os.path.join(H, "report_data.json"), encoding="utf-8"))


def slim(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: slim(v) for k, v in o.items()}
    if isinstance(o, list):
        return [slim(v) for v in o]
    return o


# drop what the page does not use (per-pair lists of the revisit evaluation)
if data.get("revisit") and "per_run" in data["revisit"]:
    data["revisit"] = {"summary": data["revisit"].get("summary")}
js = open(os.path.join(H, "report.js"), encoding="utf-8").read()
blob = json.dumps(slim(data), separators=(",", ":")).replace("</", "<\\/")
out = tpl.replace("/*DATA*/", blob).replace("/*JS*/", js)
assert "/*DATA*/" not in out and "/*JS*/" not in out
open(os.path.join(H, "index.html"), "w", encoding="utf-8", newline="\n").write(out)
print("index.html", len(out.encode("utf-8")), "bytes")
