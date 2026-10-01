#!/usr/bin/env python3
import io, os, re, zipfile, pathlib, textwrap
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageStat
import pytesseract
from pytesseract import Output
import argostranslate.package, argostranslate.translate

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANGS = ["tr","fr","it","ar","es","ja"]
ZIP_RE = re.compile(r"^\d+(?:\.\d+)?\.zip$")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_AR = "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"
FONT_JA = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

def install_models():
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    installed = {(p.from_code,p.to_code) for p in argostranslate.package.get_installed_packages()}
    for target in LANGS:
        if ("en", target) in installed: continue
        pkg = next((p for p in available if p.from_code=="en" and p.to_code==target), None)
        if not pkg:
            raise RuntimeError(f"Argos package en->{target} not found")
        argostranslate.package.install_from_path(pkg.download())

def image_names(z):
    return sorted([n for n in z.namelist() if n.lower().endswith((".webp",".png",".jpg",".jpeg"))])

def ocr_lines(img):
    data=pytesseract.image_to_data(img.convert("RGB"),lang="eng",config="--psm 11",output_type=Output.DICT)
    groups=defaultdict(list)
    for i,t in enumerate(data["text"]):
        t=(t or "").strip()
        try: conf=float(data["conf"][i])
        except: conf=-1
        if conf<45 or not t or not any(c.isalpha() for c in t): continue
        key=(data["block_num"][i],data["par_num"][i],data["line_num"][i])
        groups[key].append(i)
    out=[]
    for ids in groups.values():
        txt=" ".join(data["text"][i].strip() for i in ids if data["text"][i].strip())
        if len(txt)<2: continue
        x=min(data["left"][i] for i in ids); y=min(data["top"][i] for i in ids)
        x2=max(data["left"][i]+data["width"][i] for i in ids); y2=max(data["top"][i]+data["height"][i] for i in ids)
        out.append((txt,(x,y,x2,y2)))
    return out

def get_font(lang,size):
    p=FONT_JA if lang=="ja" else FONT_AR if lang=="ar" else FONT
    try: return ImageFont.truetype(p,max(10,size))
    except: return ImageFont.load_default()

def wrap(draw,text,font,width):
    if lang_safe(text):
        words=text.split()
    else:
        words=list(text)
    if not words: return [text]
    lines=[]; cur=words[0]
    sep=" " if lang_safe(text) else ""
    for w in words[1:]:
        trial=cur+sep+w
        if draw.textbbox((0,0),trial,font=font)[2] <= width: cur=trial
        else: lines.append(cur); cur=w
    lines.append(cur)
    return lines

def lang_safe(text):
    return " " in text or all(ord(c)<0x3000 for c in text)

def paint(img, entries, lang):
    out=img.convert("RGB")
    draw=ImageDraw.Draw(out)
    W,H=out.size
    for src,(x,y,x2,y2) in entries:
        try:
            txt=argostranslate.translate.translate(src,"en",lang).strip()
        except Exception:
            txt=src
        if not txt: continue
        bw=max(20,x2-x); bh=max(14,y2-y)
        pad=max(4,int(bh*.25))
        ex=max(0,x-pad); ey=max(0,y-pad)
        rx=min(W,x2+pad); ry=min(H,y2+pad)
        crop=out.crop((ex,ey,rx,ry)).convert("L")
        mean=ImageStat.Stat(crop).mean[0] if crop.size[0] and crop.size[1] else 255
        bg=(255,255,255) if mean>145 else (20,20,20)
        fg=(15,15,15) if mean>145 else (245,245,245)
        # Expand mostly downward/right to fit longer translations.
        rx=min(W,max(rx,ex+int(bw*1.55)))
        ry=min(H,max(ry,ey+int(bh*2.5)))
        draw.rounded_rectangle((ex,ey,rx,ry),radius=max(3,pad),fill=bg)
        size=max(10,int(bh*.95))
        font=get_font(lang,size)
        maxw=max(20,rx-ex-pad*2)
        lines=wrap(draw,txt,font,maxw)
        while len(lines)>4 and size>10:
            size-=1; font=get_font(lang,size); lines=wrap(draw,txt,font,maxw)
        lineh=max(12,draw.textbbox((0,0),"Ag",font=font)[3]+2)
        yy=ey+pad
        for row in lines[:4]:
            box=draw.textbbox((0,0),row,font=font)
            tw=box[2]-box[0]
            xx=ex+pad+max(0,(maxw-tw)//2)
            kwargs={}
            if lang=="ar": kwargs["direction"]="rtl"
            draw.text((xx,yy),row,font=font,fill=fg,**kwargs)
            yy+=lineh
            if yy>ry-pad: break
    return out

def main():
    install_models()
    originals=sorted([p for p in ROOT.glob("*.zip") if ZIP_RE.match(p.name)], key=lambda p:float(p.stem))
    color_root=ROOT/"color-source"
    sources=[(color_root/p.name if (color_root/p.name).exists() else p) for p in originals]
    print(f"Found {len(sources)} source chapter archives")
    for src in sources:
        print("Chapter",src.stem,flush=True)
        with zipfile.ZipFile(src) as zin:
            pages=[]
            for name in image_names(zin):
                raw=zin.read(name)
                img=Image.open(io.BytesIO(raw)).convert("RGB")
                pages.append((pathlib.PurePosixPath(name).name,img,ocr_lines(img)))
        for lang in LANGS:
            outdir=ROOT/lang; outdir.mkdir(exist_ok=True)
            outzip=outdir/src.name
            with zipfile.ZipFile(outzip,"w",compression=zipfile.ZIP_STORED) as zout:
                for i,(name,img,entries) in enumerate(pages,1):
                    result=paint(img,entries,lang)
                    b=io.BytesIO(); result.save(b,"WEBP",quality=88,method=4)
                    stem=pathlib.Path(name).stem
                    zout.writestr(stem+".webp",b.getvalue(),compress_type=zipfile.ZIP_STORED)
                    if i%10==0: print(f" {lang}: {i}/{len(pages)}",flush=True)
            print(" wrote",outzip.relative_to(ROOT),flush=True)

if __name__=="__main__":
    main()
