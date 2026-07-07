import ROOT
import plotly.graph_objects as go
import os
import re

# --------------------------------------------------
# Input directory
# --------------------------------------------------
input_dir = "/data/users/rserino/CROSS/PlotDataStream/"

OUTPUT_DIR = "./"
outfile = os.path.join(OUTPUT_DIR, "PlotAllRuns.html")

# --------------------------------------------------
# Find ROOT files
# --------------------------------------------------
pattern = re.compile(r"PlotDataStream_Run(\d+)\.root")

files = []
for fname in os.listdir(input_dir):
    match = pattern.match(fname)
    if match:
        run = match.group(1)
        files.append((run, os.path.join(input_dir, fname)))

# Sort by run number
files.sort(key=lambda x: int(x[0]))

print(f"Found {len(files)} runs")

# --------------------------------------------------
# Process each run
# --------------------------------------------------
run_blocks = []

for run, filepath in files:

    print(f"\nProcessing Run {run}")

    f = ROOT.TFile.Open(filepath)
    if not f or f.IsZombie():
        print(f"Skipping {filepath}")
        continue

    canvas = f.Get("datastream")
    if not canvas:
        print(f"No canvas in {filepath}")
        continue

    figures_html = []

    pads = canvas.GetListOfPrimitives()

    for pad in pads:
        if not pad.InheritsFrom("TPad"):
            continue

        primitives = pad.GetListOfPrimitives()

        for obj in primitives:
            if not obj.InheritsFrom("TMultiGraph"):
                continue

            multigraph = obj
            title = multigraph.GetTitle()  # "Channel XX"

            fig = go.Figure()

            graphs = multigraph.GetListOfGraphs()

            for g in graphs:
                x = [g.GetX()[i] for i in range(g.GetN())]
                y = [g.GetY()[i] for i in range(g.GetN())]

                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=g.GetName()
                ))

            fig.update_layout(
                title=f"<b>{title}</b>",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_white",
                height=350,
                margin=dict(t=50, b=30)
            )

            figures_html.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # Wrap this run in a DIV (hidden except first)
    run_div = f"<div class='run-block' id='run_{run}' style='display:none;'>"
    run_div += f"<h2>Run {run}</h2>"

    for fig_html in figures_html:
        run_div += "<div style='margin-bottom:40px;'>"
        run_div += fig_html
        run_div += "</div>"

    run_div += "</div>"

    run_blocks.append((run, run_div))

# --------------------------------------------------
# Write final HTML
# --------------------------------------------------
with open(outfile, "w") as fhtml:

    fhtml.write("""
<html>
<head>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script>
function showRun(run) {
    let blocks = document.getElementsByClassName("run-block");
    for (let i = 0; i < blocks.length; i++) {
        blocks[i].style.display = "none";
    }
    document.getElementById("run_" + run).style.display = "block";
}
</script>

</head>
<body>
<h1>Datastream Viewer</h1>

<select onchange="showRun(this.value)">
""")

    # Dropdown options
    for i, (run, _) in enumerate(run_blocks):
        selected = "selected" if i == 0 else ""
        fhtml.write(f"<option value='{run}' {selected}>Run {run}</option>")

    fhtml.write("</select><hr>")

    # Add run blocks
    for i, (run, block) in enumerate(run_blocks):
        if i == 0:
            block = block.replace("display:none", "display:block")
        fhtml.write(block)

    fhtml.write("""
</body>
</html>
""")

print(f"\nSaved HTML:\n  {outfile}")