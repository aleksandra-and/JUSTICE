"""
Temporary script to generate a world map coloured by RICE-12 regions.
Output: analysis/figures/rice12_regions_map.pdf  (and .png)

Requires: geopandas, matplotlib
Shapefile: downloaded once to ~/.cache/geodatasets/ne_countries/
"""

import json
import pathlib
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
REGIONS_JSON = ROOT / "data" / "input" / "rice_12_regions_dict.json"
OUT_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = pathlib.Path.home() / ".cache" / "geodatasets" / "ne_countries"
SHP = CACHE_DIR / "ne_110m_admin_0_countries.shp"

# ── download Natural Earth 110m countries shapefile if needed ─────────────────
if not SHP.exists():
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    zip_path = CACHE_DIR.parent / "ne_110m_admin_0_countries.zip"
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} …")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(CACHE_DIR)
    print("Shapefile ready.")

# ── load region mapping ────────────────────────────────────────────────────────
with open(REGIONS_JSON) as f:
    regions: dict[str, list[str]] = json.load(f)

# ── colour palette (matches the paper figure) ─────────────────────────────────
PALETTE = {
    "Africa":            "#d47b5f",  
    "Other High Income": "#090c81",   
    "India":             "#eed547",   
    "China":             "#82d1ee",   
    "Russia":            "#1C6350",  
    "United States":     "#1945e3",   
    "Japan":             "#5b31de",   
    "Latin America":     "#db9bf8",   
    "Eurasia":           "#819b65",  
    "Middle East":       "#e6a254",  
    "Europe":            "#789daa",   
    "Other Asia":        "#9fcb13",   
}

# ── build iso3 → region lookup ────────────────────────────────────────────────
iso_to_region: dict[str, str] = {}
for region, codes in regions.items():
    for iso3 in codes:
        iso_to_region[iso3] = region

# ── load Natural Earth world polygons ─────────────────────────────────────────
world = gpd.read_file(SHP)

# ISO_A3 is -99 for some territories; fall back to ADM0_A3 in those cases
world["iso3"] = world["ISO_A3"].where(world["ISO_A3"] != "-99", world["ADM0_A3"])
world["region"] = world["iso3"].map(iso_to_region)
world["color"] = world["region"].map(PALETTE).fillna("#f5f5f5")

# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_aspect("equal")
ax.axis("off")

world.plot(
    ax=ax,
    color=world["color"],
    edgecolor="white",
    linewidth=0.4,
)

# ── legend ─────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=color, label=region)
    for region, color in PALETTE.items()
]
ax.legend(
    handles=legend_handles,
    loc="lower left",
    fontsize=8,
    frameon=True,
    framealpha=0.9,
    edgecolor="grey",
    title="Region",
    title_fontsize=9,
)

plt.tight_layout()

for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"rice12_regions_map.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")

plt.show()
