import numpy as np, cv2, rig, time
from PIL import Image
S='/tmp/claude-0/-home-user-eddy/8f4112c1-3545-5e2c-9ad5-efe3ea4928c6/scratchpad'
def place(name, W=900, H=1300, mid=(450,200), io=64):
    P=rig.POSES[name]; k=io/P['io']; src=rig.load(name)
    M=np.float32([[k,0,mid[0]-k*P['mid'][0]],[0,k,mid[1]-k*P['mid'][1]]])
    return cv2.warpAffine(src,M,(W,H),flags=cv2.INTER_LINEAR)
def over_white(a): return a[...,:3]*a[...,3:]+(1-a[...,3:])
def flows(A,B):
    ga=cv2.cvtColor((over_white(A)*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    gb=cv2.cvtColor((over_white(B)*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(ga,gb,None), dis.calc(gb,ga,None)
def morph(A,B,fab,fba,t,blur=0.1,n=5):
    H,W=A.shape[:2]; gx,gy=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
    acc=0
    for ts in np.linspace(t-blur,t+blur,n) if blur>0 else [t]:
        ts=float(np.clip(ts,0,1))
        At=cv2.remap(A,gx-ts*fab[...,0],gy-ts*fab[...,1],cv2.INTER_LINEAR)
        Bt=cv2.remap(B,gx-(1-ts)*fba[...,0],gy-(1-ts)*fba[...,1],cv2.INTER_LINEAR)
        w=np.clip((ts-0.3)/0.4,0,1); w=w*w*(3-2*w)
        acc=acc+At*(1-w)+Bt*w
    return acc/(n if blur>0 else 1)
rows=[]
for a,b in [('think','point'),('calm','heart'),('think','offer')]:
    A=place(a);B=place(b); t0=time.time(); fab,fba=flows(A,B); 
    tiles=[(np.clip(over_white(morph(A,B,fab,fba,t)),0,1)*255).astype(np.uint8)[:900] for t in [0.2,0.4,0.5,0.6,0.8]]
    rows.append(np.concatenate(tiles,1)); print(a,b,time.time()-t0)
Image.fromarray(np.concatenate(rows,0)).resize((1500,900)).save(S+'/morph.png')
