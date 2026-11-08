# ===================== CELL (NEW) — M11: stress-test realism & statistical power =====================
# A) Real high-latency benign flows (no synthesis): FGSAN false-positive rate on the genuinely slow
#    tail of real benign flows, raw counts + Wilson CIs, vs fast benign.
# B) Physically-consistent synthetic stress: stretch IAT/idle/payload then RECOMPUTE the rate features
#    from the primitives so each stressed flow is realizable; audit feasibility; raw FP counts + Wilson
#    CI; paired McNemar (discordances b,c + exact p) vs the same flows' clean predictions.
# C) Multiple seeds: repeat (B) over 5 seeds; per-seed raw counts, mean+/-sd, pooled exact McNemar.
# Needs restored: X_test_raw, y_test, scaler, present_features, main_model; defined: predict_torch.
import os, json, numpy as np
from math import sqrt
from scipy.stats import binomtest

try:
    from google.colab import drive; drive.mount("/content/drive")
    OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception:
    OUT="./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True)
RES=f"{OUT}/m11_results.json"; R={}

FEAT=list(present_features); fi={n:i for i,n in enumerate(FEAT)}
def col(name): return fi[name]
US=1e6
I_DUR=col("Flow Duration")
I_TLF=col("Total Length of Fwd Packets"); I_TLB=col("Total Length of Bwd Packets")
I_PF =col("Total Fwd Packets");           I_PB =col("Total Backward Packets")
I_BPS=col("Flow Bytes/s");                I_PPS=col("Flow Packets/s")
IAT_COLS =[c for c in FEAT if "IAT" in c or c=="Flow Duration"]
IDLE_COLS=[c for c in FEAT if "Idle" in c]
LEN_COLS =[c for c in FEAT if "Length" in c]      # Total Length + Packet Length Mean

def wilson(k,n,z=1.96):
    if n==0: return (float("nan"),float("nan"))
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h),min(1,c+h))

def fpr_block(model,Xstd,tag):
    preds,_=predict_torch(model,Xstd); preds=np.asarray(preds)
    k=int((preds==1).sum()); n=len(preds); lo,hi=wilson(k,n)
    print(f"  {tag:32} FP={k:4d}/{n:<5d}  FPR={k/n:.4f}  95%CI[{lo:.4f},{hi:.4f}]")
    return dict(fp=k,n=n,fpr=k/n,ci=[lo,hi],preds=preds)

def recompute_rates(raw):
    r=raw.copy()
    dur=np.maximum(r[:,I_DUR],1e-9)
    r[:,I_BPS]=(r[:,I_TLF]+r[:,I_TLB])*US/dur
    r[:,I_PPS]=(r[:,I_PF]+r[:,I_PB])*US/dur
    return r

def phys_stretch(raw, idx, iat_m, idle_m, len_m):
    r=raw.copy()
    for j,i in enumerate(idx):
        for c in IAT_COLS:  r[i,fi[c]]*=iat_m[j]
        for c in IDLE_COLS: r[i,fi[c]]*=idle_m[j]
        for c in LEN_COLS:  r[i,fi[c]]*=len_m[j]
    return recompute_rates(r)                      # rates made exact -> realizable

def feas_violation(raw):
    dur=np.maximum(raw[:,I_DUR],1e-9)
    bps=(raw[:,I_TLF]+raw[:,I_TLB])*US/dur; pps=(raw[:,I_PF]+raw[:,I_PB])*US/dur
    bad=(np.abs(bps-raw[:,I_BPS])>1e-3*np.maximum(np.abs(raw[:,I_BPS]),1)) | \
        (np.abs(pps-raw[:,I_PPS])>1e-3*np.maximum(np.abs(raw[:,I_PPS]),1)) | \
        (raw<-1e-6).any(axis=1)
    return float(bad.mean())

ben = y_test==0
Xb_raw = X_test_raw[ben]; nben=len(Xb_raw)
dur_b = Xb_raw[:,I_DUR]
print(f"benign test flows: {nben}")

# ---------- Part A: REAL high-latency benign flows ----------
print("\n[A] Real high-latency benign flows (no synthesis)")
order=np.argsort(dur_b)
for frac,label in [(0.25,"slowest 25% (real)"),(0.10,"slowest 10% (real)")]:
    k=int(nben*frac); slow=order[-k:]; fast=order[:nben-k]
    bs=fpr_block(main_model, scaler.transform(Xb_raw[slow]), label)
    R[f"real_slow_{int(frac*100)}"]={"fp":bs["fp"],"n":bs["n"],"fpr":bs["fpr"],"ci":bs["ci"]}
fs=fpr_block(main_model, scaler.transform(Xb_raw[order[:int(nben*0.75)]]), "fastest 75% (real, reference)")
R["real_fast_75"]={"fp":fs["fp"],"n":fs["n"],"fpr":fs["fpr"],"ci":fs["ci"]}

# ---------- Part B: physically-consistent synthetic stress (single seed) ----------
print("\n[B] Physically-consistent synthetic stress (recomputed rates), seed 42")
SEED=42; rng=np.random.default_rng(SEED)
frac_stress=0.20; ns=int(nben*frac_stress)
sidx=rng.choice(nben,size=ns,replace=False)
iat_m =rng.uniform(1.5,4.0,ns); idle_m=rng.uniform(1.5,3.0,ns); len_m=rng.uniform(0.8,1.3,ns)
Xb_stress=phys_stretch(Xb_raw, sidx, iat_m, idle_m, len_m)
print(f"  feasibility violations after recompute: {feas_violation(Xb_stress[sidx]):.4%}")
clean_pred,_=predict_torch(main_model, scaler.transform(Xb_raw[sidx]));  clean_pred=np.asarray(clean_pred)
str_pred,_ =predict_torch(main_model, scaler.transform(Xb_stress[sidx]));str_pred=np.asarray(str_pred)
kc,ks=int((clean_pred==1).sum()),int((str_pred==1).sum())
lo_c,hi_c=wilson(kc,ns); lo_s,hi_s=wilson(ks,ns)
b=int(((clean_pred==0)&(str_pred==1)).sum())   # became FP under stress
c=int(((clean_pred==1)&(str_pred==0)).sum())   # FP cleared under stress
p=binomtest(b, b+c, 0.5).pvalue if (b+c)>0 else 1.0
print(f"  clean  FP={kc}/{ns}  FPR={kc/ns:.4f} CI[{lo_c:.4f},{hi_c:.4f}]")
print(f"  stress FP={ks}/{ns}  FPR={ks/ns:.4f} CI[{lo_s:.4f},{hi_s:.4f}]")
print(f"  McNemar discordances: b(clean->FP)={b}  c(FP->clean)={c}  exact p={p:.4f}")
R["stressB"]={"ns":ns,"clean_fp":kc,"stress_fp":ks,"clean_ci":[lo_c,hi_c],"stress_ci":[lo_s,hi_s],
              "b":b,"c":c,"p":p,"feas_viol":feas_violation(Xb_stress[sidx])}

# ---------- Part C: multiple seeds ----------
print("\n[C] Multiple seeds (physically-consistent stress)")
per=[]; B=C=0
for s in range(5):
    rng=np.random.default_rng(100+s)
    sidx=rng.choice(nben,size=ns,replace=False)
    iat_m=rng.uniform(1.5,4.0,ns); idle_m=rng.uniform(1.5,3.0,ns); len_m=rng.uniform(0.8,1.3,ns)
    Xs=phys_stretch(Xb_raw,sidx,iat_m,idle_m,len_m)
    cp,_=predict_torch(main_model,scaler.transform(Xb_raw[sidx])); cp=np.asarray(cp)
    sp,_=predict_torch(main_model,scaler.transform(Xs[sidx]));     sp=np.asarray(sp)
    ks=int((sp==1).sum()); bb=int(((cp==0)&(sp==1)).sum()); cc=int(((cp==1)&(sp==0)).sum())
    per.append(ks/ns); B+=bb; C+=cc
    print(f"  seed {s}: stress FP={ks}/{ns}  FPR={ks/ns:.4f}  b={bb} c={cc}")
per=np.array(per); pooled_p=binomtest(B,B+C,0.5).pvalue if (B+C)>0 else 1.0
print(f"  stress FPR across seeds: mean={per.mean():.4f} sd={per.std(ddof=1):.4f} range=[{per.min():.4f},{per.max():.4f}]")
print(f"  pooled McNemar: b={B} c={C} exact p={pooled_p:.4f}")
R["stressC"]={"fpr_mean":float(per.mean()),"fpr_sd":float(per.std(ddof=1)),
              "fpr_range":[float(per.min()),float(per.max())],"B":B,"C":C,"pooled_p":pooled_p,"ns":ns}
json.dump(R,open(RES,"w"),indent=1)
print("\nsaved ->",RES)
