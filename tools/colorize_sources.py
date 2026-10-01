#!/usr/bin/env python3
import io, os, re, sys, zipfile, pathlib, tempfile, subprocess, shutil
from PIL import Image

ROOT=pathlib.Path(__file__).resolve().parents[1]
COLORIZER=pathlib.Path(os.environ.get("MANGA_COLORIZER","/tmp/manga-colorization-v2"))
ZIP_RE=re.compile(r"^\d+(?:\.\d+)?\.zip$")

def pages(z):
    return sorted([n for n in z.namelist() if n.lower().endswith((".webp",".png",".jpg",".jpeg"))])

def main():
    outroot=ROOT/"color-source"
    outroot.mkdir(exist_ok=True)
    sources=sorted([p for p in ROOT.glob("*.zip") if ZIP_RE.match(p.name)],key=lambda p:float(p.stem))
    print("Colorizing",len(sources),"chapters",flush=True)
    for src in sources:
        outzip=outroot/src.name
        print("Chapter",src.stem,flush=True)
        with tempfile.TemporaryDirectory() as td:
            td=pathlib.Path(td); inp=td/"input"; inp.mkdir()
            with zipfile.ZipFile(src) as zin:
                names=pages(zin)
                for i,name in enumerate(names,1):
                    img=Image.open(io.BytesIO(zin.read(name))).convert("RGB")
                    img.save(inp/f"{i:03d}.png","PNG")
            cmd=[sys.executable,str(COLORIZER/"inference.py"),"-p",str(inp),"-nd","-s","576"]
            subprocess.run(cmd,cwd=COLORIZER,check=True)
            color=inp/"colorization"
            files=sorted(color.glob("*.png"))
            if len(files)!=len(names):
                raise RuntimeError(f"{src.name}: expected {len(names)} outputs, got {len(files)}")
            with zipfile.ZipFile(outzip,"w",compression=zipfile.ZIP_STORED) as zout:
                for i,p in enumerate(files,1):
                    img=Image.open(p).convert("RGB")
                    b=io.BytesIO(); img.save(b,"WEBP",quality=88,method=4)
                    zout.writestr(f"{i:03d}.webp",b.getvalue(),compress_type=zipfile.ZIP_STORED)
            print(" wrote",outzip.relative_to(ROOT),flush=True)

if __name__=="__main__":
    main()
