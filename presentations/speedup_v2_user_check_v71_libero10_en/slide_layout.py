"""Small vector PDF slide layout helpers. No policy/simulator dependencies."""
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader

W, H = 960, 540
NAVY = '#13283F'
TEAL = '#007F87'
BLUE = '#3E69A8'
GREEN = '#218261'
RED = '#BF4651'
AMBER = '#BD7916'
MUTED = '#536A7D'
LIGHT = '#EFF4F8'
WHITE = '#FFFFFF'
INK = '#172E45'

FONT = Path('/usr/share/fonts/truetype/dejavu')
for name, file in [('Regular','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),
                   ('Italic','DejaVuSans-Oblique.ttf'),('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT/file)))
pdfmetrics.registerFontFamily('Regular',normal='Regular',bold='Bold',italic='Italic',boldItalic='Bold')

class Deck:
    def __init__(self, path):
        self.c=canvas.Canvas(str(path),pagesize=(W,H),pageCompression=1)
        self.c.setTitle('SpeedUp v2 User Check 7.1: Technical Framework on LIBERO-10')
        self.c.setAuthor('gpt_field project')
        self.c.setSubject('Frozen UC7.1 architecture, browser approval, calibrated direction semantics, measured motion and audited t0/t1 results; October 30, 2026')
        self.pages=[]; self.n=0; self.dark=False; self.bounds=[]

    def rect(self,x,y,w,h,color=LIGHT,r=10,stroke=None):
        self.c.setFillColor(HexColor(color))
        if stroke: self.c.setStrokeColor(HexColor(stroke))
        self.c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=bool(stroke))

    def line(self,x1,y1,x2,y2,color=MUTED,width=1):
        self.c.setStrokeColor(HexColor(color));self.c.setLineWidth(width)
        self.c.line(x1,H-y1,x2,H-y2)

    def arrow(self,x1,y1,x2,y2,color=TEAL,width=2):
        import math
        self.line(x1,y1,x2,y2,color,width)
        angle=math.atan2(y2-y1,x2-x1)
        for sign in [-1,1]:
            self.line(x2,y2,x2-7*math.cos(angle+sign*.45),y2-7*math.sin(angle+sign*.45),color,width)

    def text(self,x,y,text,size=16,color=INK,font='Regular'):
        self.c.setFillColor(HexColor(color));self.c.setFont(font,size)
        self.c.drawString(x,H-y-size*.84,str(text))

    def p(self,x,y,w,text,size=16,color=INK,leading=None,font='Regular',max_h=None):
        style=ParagraphStyle('p',fontName=font,fontSize=size,leading=leading or size*1.38,
                             textColor=HexColor(color),spaceAfter=0,allowWidows=0,allowOrphans=0)
        p=Paragraph(text,style); aw,ah=p.wrap(w,H)
        if max_h is not None and ah > max_h+.5:
            raise ValueError(f'Page {self.n}: paragraph {ah:.1f}>{max_h}: {text[:100]}')
        if y+ah>499 and self.n:
            raise ValueError(f'Page {self.n}: body exceeds footer ({y+ah:.1f}): {text[:100]}')
        p.drawOn(self.c,x,H-y-ah)
        self.bounds.append({'page':self.n,'x':x,'y':y,'w':w,'h':ah,'text':text})
        return ah

    def image(self,path,x,y,w,h):
        from PIL import Image
        with Image.open(path) as im: iw,ih=im.size
        ratio=min(w/iw,h/ih); dw,dh=iw*ratio,ih*ratio
        self.c.drawImage(str(path),x+(w-dw)/2,H-y-dh-(h-dh)/2,dw,dh,mask='auto')

    def begin(self,title,kicker='METHOD',subtitle=None,dark=False):
        self.n+=1; self.dark=dark
        self.pages.append({'page':self.n,'title':title})
        self.c.bookmarkPage('page_'+str(self.n))
        self.c.addOutlineEntry(title,'page_'+str(self.n),level=0,closed=False)
        self.rect(0,0,W,H,NAVY if dark else '#FAFCFE',r=0)
        self.rect(36,32,5,28,TEAL,r=1)
        self.text(51,32,kicker.upper(),10,'#77D7D1' if dark else TEAL,'Bold')
        title_size=min(29,29*888/max(888,pdfmetrics.stringWidth(title,'Bold',29)))
        self.text(36,64,title,title_size,WHITE if dark else INK,'Bold')
        if subtitle: self.p(37,105,884,subtitle,12,'#BDCCD9' if dark else MUTED,max_h=35)

    def footer(self,source):
        self.line(36,506,924,506,'#365067' if self.dark else '#D8E2EA',.7)
        self.text(36,516,source,7.0,'#AEC3D4' if self.dark else MUTED)
        self.text(874,515,f'{self.n:02d}',10,'#AEC3D4' if self.dark else MUTED,'Bold')
        self.c.showPage()

    def card(self,x,y,w,h,title,body,color=TEAL,size=16):
        self.rect(x,y,w,h,LIGHT)
        self.rect(x,y,4,h,color,r=0)
        self.p(x+18,y+17,w-36,title,17,color,font='Bold',max_h=55)
        self.p(x+18,y+60,w-36,body,size,max_h=h-72)

    def metric(self,x,y,w,value,label,color=TEAL):
        self.rect(x,y,w,105,LIGHT)
        self.text(x+18,y+17,value,31,color,'Bold')
        self.p(x+18,y+62,w-36,label,12,MUTED,max_h=37)

    def bullet(self,x,y,w,title,body,color=TEAL,size=16):
        self.rect(x,y+6,6,6,color,r=3)
        a=self.p(x+19,y,w-19,f'<b>{title}</b>',size,INK)
        b=self.p(x+19,y+a+4,w-19,body,size-1,MUTED)
        return a+b+22

    def table(self,x,y,widths,headers,rows,row_h=56,font=13,colors=None):
        total=sum(widths); self.rect(x,y,total,35,NAVY,r=4)
        xx=x
        for w,v in zip(widths,headers):
            self.p(xx+10,y+8,w-20,escape(str(v)),11,WHITE,font='Bold',max_h=26);xx+=w
        yy=y+35
        for i,row in enumerate(rows):
            self.rect(x,yy,total,row_h,WHITE if i%2 else LIGHT,r=0)
            xx=x
            for j,(w,v) in enumerate(zip(widths,row)):
                color=(colors or {}).get((i,j),INK)
                self.p(xx+10,yy+8,w-20,str(v),font,color,leading=font*1.25,max_h=row_h-12);xx+=w
            yy+=row_h
        return yy

    def save(self):
        self.c.save()
