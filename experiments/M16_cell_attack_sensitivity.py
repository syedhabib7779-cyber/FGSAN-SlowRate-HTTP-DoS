# ===================== CELL (NEW) — M16: attack-parameter sensitivity with uncertainty =====================
# ESR vs perturbation budget epsilon, PGD steps, and random restarts, each as mean +/- 95% CI over seeds,
# under the paper's feature-space L-inf threat model (consistent constraints). Random-restart PGD with random
# initialization is used so the curves carry seed uncertainty. Denominator = the attack flows the model
# classifies correctly on clean input. Needs restored: X_atk, y_atk, main_model; defined: predict_torch.
import os, json, numpy as np, torch
import torch.nn.functional as F
try:
    from google.colab import drive; drive.mount("/content/drive"); OUT="/content/drive/MyDrive/fgsan_ckpt"
except Exception: OUT="./fgsan_ckpt"
os.makedirs(OUT, exist_ok=True); RES=f"{OUT}/m16_results.json"; R={}
DEV=globals().get("DEVICE","cuda" if torch.cuda.is_available() else "cpu")
main_model.eval()

Xa=np.asarray(X_atk,dtype=np.float32)
cleanp,_=predict_torch(main_model,Xa); cc=(np.asarray(cleanp)==1)   # clean-correct attack flows
denom=int(cc.sum())
x0=torch.tensor(Xa,dtype=torch.float32,device=DEV)
ytrue=torch.ones(len(Xa),dtype=torch.long,device=DEV)
print(f"attack flows {len(Xa)}; clean-correct denominator {denom}")

def pgd(eps,alpha,steps,seed):
    g=torch.Generator(device=DEV).manual_seed(int(seed))
    delta=((torch.rand(x0.shape,generator=g,device=DEV)*2-1)*eps)
    for _ in range(steps):
        x=(x0+delta).detach().requires_grad_(True)
        loss=F.cross_entropy(main_model(x)[0],ytrue)
        grad=torch.autograd.grad(loss,x)[0]
        with torch.no_grad():
            delta=torch.clamp(delta+alpha*grad.sign(),-eps,eps)
    return (x0+delta).detach().cpu().numpy()

def esr_of(Xadv):
    p,_=predict_torch(main_model,Xadv); p=np.asarray(p)
    return float((cc&(p==0)).sum()/denom)

def ci(vals):
    v=np.array(vals); m=v.mean(); sd=v.std(ddof=1) if len(v)>1 else 0.0
    h=1.96*sd/np.sqrt(len(v)); return m,sd,max(0,m-h),m+h

SEEDS=[0,1,2,3,4]
def sweep(label, configs, make):
    print(f"\n=== {label} ===")
    print(f"{'setting':>12} {'ESR mean':>9} {'sd':>7} {'95% CI':>20}")
    out=[]
    for cfgname,fn in configs:
        vals=[esr_of(make(fn,s)) for s in SEEDS]
        m,sd,lo,hi=ci(vals)
        out.append({"setting":cfgname,"mean":m,"sd":sd,"ci":[lo,hi]})
        print(f"{cfgname:>12} {m:9.4f} {sd:7.4f}   [{lo:.4f}, {hi:.4f}]")
    return out

# 1) epsilon sweep (steps=20, alpha=2.5*eps/steps)
R["epsilon"]=sweep("ESR vs perturbation budget epsilon (steps=20)",
    [(f"{e:.2f}", e) for e in [0.01,0.02,0.05,0.10,0.20]],
    lambda e,s: pgd(e, 2.5*e/20, 20, s))
# 2) steps sweep (eps=0.05)
R["steps"]=sweep("ESR vs PGD steps (epsilon=0.05)",
    [(str(k), k) for k in [1,7,20,50,100]],
    lambda k,s: pgd(0.05, 2.5*0.05/k, k, s))
# 3) restarts sweep (eps=0.05, steps=20, best-of-R); uncertainty over 3 seed groups
print("\n=== ESR vs random restarts (epsilon=0.05, steps=20, best-of-R) ===")
print(f"{'restarts':>9} {'ESR mean':>9} {'sd':>7} {'95% CI':>20}")
R["restarts"]=[]
for Rn in [1,2,5,10,20]:
    vals=[]
    for grp in [0,100,200]:
        evaded=np.zeros(len(Xa),bool)
        for r in range(Rn):
            p,_=predict_torch(main_model,pgd(0.05,2.5*0.05/20,20,grp*10+r)); p=np.asarray(p)
            evaded|=(cc&(p==0))
        vals.append(evaded.sum()/denom)
    m,sd,lo,hi=ci(vals)
    R["restarts"].append({"restarts":Rn,"mean":m,"sd":sd,"ci":[lo,hi]})
    print(f"{Rn:9d} {m:9.4f} {sd:7.4f}   [{lo:.4f}, {hi:.4f}]")

json.dump(R,open(RES,"w"),indent=1)
print("\nReading: a flat ESR-vs-steps and ESR-vs-restarts curve => the attack has converged and more compute")
print("does not find more evasions (no gradient masking); ESR rises smoothly with epsilon as expected.")
print("saved ->",RES)
