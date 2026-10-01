#!/usr/bin/env python3
import io, os, re, zipfile, pathlib
from collections import defaultdict, Counter
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageStat
import pytesseract
from pytesseract import Output
import argostranslate.package, argostranslate.translate

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANGS = [x.strip() for x in os.environ.get("TARGET_LANGS","tr").split(",") if x.strip()]
ZIP_RE = re.compile(r"^\d+(?:\.\d+)?\.zip$")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def install_models():
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    installed = {(p.from_code,p.to_code) for p in argostranslate.package.get_installed_packages()}
    for target in LANGS:
        if ("en", target) in installed:
            continue
        pkg = next((p for p in available if p.from_code=="en" and p.to_code==target), None)
        if not pkg:
            raise RuntimeError(f"Argos package en->{target} not found")
        argostranslate.package.install_from_path(pkg.download())

def image_names(z):
    return sorted([n for n in z.namelist() if n.lower().endswith((".webp",".png",".jpg",".jpeg"))])

def ocr_words(img):
    data = pytesseract.image_to_data(img.convert("RGB"), lang="eng", config="--psm 11", output_type=Output.DICT)
    words=[]
    for i,t in enumerate(data["text"]):
        t=(t or "").strip()
        try: conf=float(data["conf"][i])
        except: conf=-1
        if conf < 48 or not t or not any(c.isalpha() for c in t):
            continue
        x,y,w,h=[int(data[k][i]) for k in ("left","top","width","height")]
        if w<2 or h<4:
            continue
        words.append({
            "text":t,"x":x,"y":y,"w":w,"h":h,
            "block":int(data["block_num"][i]),
            "par":int(data["par_num"][i]),
            "line":int(data["line_num"][i]),
        })
    return words

def bright_components(img):
    arr=np.asarray(img.convert("RGB"))
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    # Speech balloons / caption boxes are normally light. Closing joins white around letters.
    bright=(gray>=205).astype(np.uint8)*255
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    bright=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,kernel,iterations=2)
    n,labels,stats,_=cv2.connectedComponentsWithStats(bright,8)
    return gray,labels,stats

def label_for_word(word,labels,stats):
    H,W=labels.shape
    x,y,w,h=word["x"],word["y"],word["w"],word["h"]
    pts=[
      (x-3,y+h//2),(x+w+3,y+h//2),(x+w//2,y-3),(x+w//2,y+h+3),
      (x-2,y-2),(x+w+2,y-2),(x-2,y+h+2),(x+w+2,y+h+2)
    ]
    labs=[]
    for px,py in pts:
        px=max(0,min(W-1,px)); py=max(0,min(H-1,py))
        lab=int(labels[py,px])
        if lab>0: labs.append(lab)
    if not labs:
        return 0
    lab=Counter(labs).most_common(1)[0][0]
    area=int(stats[lab,cv2.CC_STAT_AREA])
    # Reject giant page backgrounds and tiny specks.
    if area < max(120,w*h*2) or area > W*H*0.42:
        return 0
    return lab

def build_groups(img):
    words=ocr_words(img)
    gray,labels,stats=bright_components(img)
    grouped=defaultdict(list)
    for w in words:
        lab=label_for_word(w,labels,stats)
        if not lab:
            continue
        grouped[lab].append(w)

    groups=[]
    for lab,ws in grouped.items():
        # Sort in normal reading order.
        ws=sorted(ws,key=lambda q:(q["y"],q["x"]))
        text=" ".join(w["text"] for w in ws)
        if len(text)<2:
            continue
        x=min(w["x"] for w in ws); y=min(w["y"] for w in ws)
        x2=max(w["x"]+w["w"] for w in ws); y2=max(w["y"]+w["h"] for w in ws)
        bx=int(stats[lab,cv2.CC_STAT_LEFT]); by=int(stats[lab,cv2.CC_STAT_TOP])
        bw=int(stats[lab,cv2.CC_STAT_WIDTH]); bh=int(stats[lab,cv2.CC_STAT_HEIGHT])
        # Text region can expand a little, but must stay inside the original light region.
        padx=max(5,int((x2-x)*0.10)); pady=max(4,int((y2-y)*0.18))
        tx1=max(bx+3,x-padx); ty1=max(by+3,y-pady)
        tx2=min(bx+bw-3,x2+padx); ty2=min(by+bh-3,y2+pady)
        if tx2-tx1<18 or ty2-ty1<10:
            continue
        # Only accept regions that are genuinely light around the text.
        crop=gray[max(0,ty1):min(gray.shape[0],ty2),max(0,tx1):min(gray.shape[1],tx2)]
        if crop.size==0 or float((crop>190).mean()) < 0.58:
            continue
        groups.append({"label":lab,"words":ws,"text":text,"box":(tx1,ty1,tx2,ty2)})
    return groups

def get_font(size):
    try: return ImageFont.truetype(FONT,max(8,size))
    except: return ImageFont.load_default()

def wrap_text(draw,text,font,maxw):
    words=text.split()
    if not words:return [text]
    lines=[]; cur=words[0]
    for w in words[1:]:
        trial=cur+" "+w
        if draw.textbbox((0,0),trial,font=font)[2] <= maxw:
            cur=trial
        else:
            lines.append(cur);cur=w
    lines.append(cur)
    return lines

def fit_text(draw,text,box):
    x1,y1,x2,y2=box
    maxw=max(12,x2-x1); maxh=max(10,y2-y1)
    size=max(9,min(34,int(maxh*0.62)))
    while size>=8:
        font=get_font(size)
        lines=wrap_text(draw,text,font,maxw)
        sample=draw.textbbox((0,0),"Ag",font=font)
        lineh=max(9,(sample[3]-sample[1])+2)
        total=lineh*len(lines)
        widest=max((draw.textbbox((0,0),ln,font=font)[2] for ln in lines),default=0)
        if total<=maxh and widest<=maxw:
            return font,lines,lineh
        size-=1
    font=get_font(8)
    return font,wrap_text(draw,text,font,maxw),10

def erase_original_text(img,groups):
    arr=cv2.cvtColor(np.asarray(img.convert("RGB")),cv2.COLOR_RGB2BGR)
    gray=cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
    mask=np.zeros(gray.shape,np.uint8)
    H,W=gray.shape
    for g in groups:
        for w in g["words"]:
            x,y,ww,hh=w["x"],w["y"],w["w"],w["h"]
            p=max(1,int(hh*0.10))
            x1=max(0,x-p);y1=max(0,y-p);x2=min(W,x+ww+p);y2=min(H,y+hh+p)
            roi=gray[y1:y2,x1:x2]
            # Mask only dark glyph pixels, not the whole rectangle.
            local=(roi<175).astype(np.uint8)*255
            local=cv2.dilate(local,np.ones((2,2),np.uint8),iterations=1)
            mask[y1:y2,x1:x2]=np.maximum(mask[y1:y2,x1:x2],local)
    if mask.max()==0:
        return img.convert("RGB")
    cleaned=cv2.inpaint(arr,mask,3,cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(cleaned,cv2.COLOR_BGR2RGB))

def paint(img,groups,lang):
    out=erase_original_text(img,groups)
    draw=ImageDraw.Draw(out)
    for g in groups:
        try:
            txt=argostranslate.translate.translate(g["text"],"en",lang).strip()
        except Exception:
            txt=g["text"]
        if not txt:
            continue
        x1,y1,x2,y2=g["box"]
        font,lines,lineh=fit_text(draw,txt,(x1,y1,x2,y2))
        total=lineh*len(lines)
        yy=y1+max(0,((y2-y1)-total)//2)
        for line in lines:
            box=draw.textbbox((0,0),line,font=font)
            tw=box[2]-box[0]
            xx=x1+max(0,((x2-x1)-tw)//2)
            draw.text((xx,yy),line,font=font,fill=(15,15,15))
            yy+=lineh
            if yy>y2:
                break
    return out

def main():
    install_models()
    originals=sorted([p for p in ROOT.glob("*.zip") if ZIP_RE.match(p.name)],key=lambda p:float(p.stem))
    selected={x.strip() for x in os.environ.get("TARGET_CHAPTERS","").split(",") if x.strip()}
    if selected:
        originals=[p for p in originals if p.stem in selected]
    print(f"Found {len(originals)} source chapter archives")
    for src in originals:
        print("Chapter",src.stem,flush=True)
        with zipfile.ZipFile(src) as zin:
            pages=[]
            for name in image_names(zin):
                raw=zin.read(name)
                img=Image.open(io.BytesIO(raw)).convert("RGB")
                pages.append((pathlib.PurePosixPath(name).name,img,build_groups(img)))
        for lang in LANGS:
            outdir=ROOT/("tr-clean" if lang=="tr" else lang)
            outdir.mkdir(exist_ok=True)
            outzip=outdir/src.name
            with zipfile.ZipFile(outzip,"w",compression=zipfile.ZIP_STORED) as zout:
                for i,(name,img,groups) in enumerate(pages,1):
                    result=paint(img,groups,lang)
                    b=io.BytesIO();result.save(b,"WEBP",quality=90,method=4)
                    zout.writestr(pathlib.Path(name).stem+".webp",b.getvalue(),compress_type=zipfile.ZIP_STORED)
                    if i%10==0:
                        print(f" {lang}: {i}/{len(pages)}",flush=True)
            print(" wrote",outzip.relative_to(ROOT),flush=True)

if __name__=="__main__":
    main()
