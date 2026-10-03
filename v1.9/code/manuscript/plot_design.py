"""Experimental design diagram; all counts are bound to saved protocols."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

OUT = Path(__file__).resolve().parent
D18 = OUT.parent
plt.rcParams.update({"font.family":"Arial", "font.size":8, "pdf.fonttype":42, "ps.fonttype":42})


def box(ax, xy, wh, title, detail, fill="#F1F5F8", edge="#657C8B"):
    x,y = xy; w,h = wh
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.009",
                               linewidth=.8,edgecolor=edge,facecolor=fill))
    ax.text(x+w/2,y+h*.70,title,ha="center",va="center",fontweight="bold",fontsize=8.5,color="#183B50")
    ax.text(x+w/2,y+h*.32,detail,ha="center",va="center",fontsize=7.8,linespacing=1.35)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=9,linewidth=.8,color="#516671"))


fig,ax = plt.subplots(figsize=(174/25.4,118/25.4))
fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
box(ax,(.10,.862),(.80,.11),"Common zero-exposure training background (q0)",
    "Onion: 240 sources / 480 images     Potato: 268 leaves / 536 images")
box(ax,(.035,.623),(.435,.16),"Joint replacement",
    "48 onion or 54 potato probe sources per allocation\n30 splits; five complementary allocation pairs")
box(ax,(.530,.623),(.435,.16),"Single-source replacement",
    "One paired probe source at a time\nFirst 10 splits; every paired probe retained")
arrow(ax,(.30,.853),(.25,.793)); arrow(ax,(.70,.853),(.75,.793))
box(ax,(.10,.457),(.80,.095),"Within either design: identical donor deletion",
    "E and H remove the same mapped source(s); the remaining background is shared",fill="#EDF1F2")
arrow(ax,(.25,.613),(.37,.563)); arrow(ax,(.75,.613),(.63,.563))
box(ax,(.035,.219),(.435,.16),"Related-view exposure (E)",
    "Insert two non-anchor views\nfrom each selected probe source",fill="#EAF1F6",edge="#173D59")
box(ax,(.530,.219),(.435,.16),"Ordinary replacement (H)",
    "Insert two views from each reserved source\nwith the same original class",fill="#FAF2E9",edge="#AD5B16")
arrow(ax,(.35,.446),(.25,.390)); arrow(ax,(.65,.446),(.75,.390))
box(ax,(.10,.038),(.80,.102),"Evaluate the same fixed anchors in both arms",
    "Probe anchors held out; sentinels never exposed\nIdentical training-image counts and original-class support",fill="#F5F7F8")
arrow(ax,(.25,.209),(.37,.151)); arrow(ax,(.75,.209),(.63,.151))
for ext in ("pdf","eps","png"):
    fig.savefig(OUT/f"Fig_1_Matched_Replacement_Design.{ext}",dpi=600,facecolor="white")
plt.close(fig)
p=OUT/"Fig_1_Matched_Replacement_Design.png"
with Image.open(p) as image:
    image.convert("RGB").save(p,dpi=(600,600))
metadata=dict(width_mm=174,height_mm=118,counts=dict(onion=dict(training_sources=240,training_images=480,joint_selected=48,single_selected=1),
              potato=dict(training_sources=268,training_images=536,joint_selected=54,single_selected=1)),
              joint_splits=30,joint_pairs=5,single_splits=10,
              scope="Schematic of frozen designs; identical background refers to each matched E/H pair. Background is shared for single-source E/H; in joint E/H all selected replacement slots jointly differ.",
              source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/"Fig_1_Design_Source.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
(OUT/"FIG1_CAPTION_EN.txt").write_text("Fig. 1 Matched replacement at two intervention extents. Each E/H pair deletes identical donors and inserts two views per replaced source with the same original class. The joint design changes all selected sources simultaneously; the single-source design changes only the selected target's insertion at the common zero-exposure background. Anchors, image budget and class support remain fixed within each pair. Source isolation refers to audited onion components or author-recorded potato leaves, not verified whole-plant independence.\n",encoding="utf-8")
print("Design figure exported.")
