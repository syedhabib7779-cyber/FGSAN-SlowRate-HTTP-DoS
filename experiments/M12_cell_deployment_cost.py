# ===================== CELL (NEW) — M12: empirical deployment cost & time-to-detection =====================
# Measures (A) model inference latency / throughput / memory on CPU, incl. single-core (edge proxy), and
# (B) the deployment reality: the 27 features are flow-terminal aggregates, so time-to-detection is bounded
# by flow completion, not model latency -- shown with the real attack-flow duration distribution.
# Needs restored: main_model, X_test, X_test_raw, y_test, present_features.
import os, json, time, copy, numpy as np, torch

OUT=globals().get("OUT", "/content/drive/MyDrive/fgsan_ckpt" if os.path.isdir("/content/drive/MyDrive") else "./fgsan_ckpt")
os.makedirs(OUT, exist_ok=True); RES=f"{OUT}/m12_results.json"; R={}

m_cpu=copy.deepcopy(main_model).to("cpu").eval()
nparam=sum(p.numel() for p in m_cpu.parameters())
pbytes=sum(p.numel()*p.element_size() for p in m_cpu.parameters())
import io
buf=io.BytesIO(); torch.save(m_cpu.state_dict(), buf); disk=buf.getbuffer().nbytes
print(f"parameters: {nparam:,}   parameter memory: {pbytes/1024:.1f} KB   serialized state_dict: {disk/1024:.1f} KB")
R["params"]=int(nparam); R["param_kb"]=pbytes/1024; R["disk_kb"]=disk/1024

X=np.asarray(X_test, dtype=np.float32)
DEFAULT_THREADS=torch.get_num_threads()

def bench(nthreads, batch, n_iter):
    torch.set_num_threads(nthreads)
    xb=torch.tensor(X[:batch], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(10): m_cpu(xb)                 # warm-up
        t=np.empty(n_iter)
        for i in range(n_iter):
            s=time.perf_counter(); m_cpu(xb); t[i]=time.perf_counter()-s
    return t*1e3                                       # ms per batch

print("\n[A1] Single-flow latency (batch=1), CPU")
for nt,label in [(1,"1 core (edge proxy)"), (DEFAULT_THREADS,f"{DEFAULT_THREADS} cores")]:
    t=bench(nt,1,1000)
    row=dict(mean=float(t.mean()),median=float(np.median(t)),p95=float(np.percentile(t,95)),
             p99=float(np.percentile(t,99)),flows_per_s=float(1000.0/t.mean()))
    R[f"lat_b1_{nt}c"]=row
    print(f"  {label:18} ms: mean={row['mean']:.3f} med={row['median']:.3f} "
          f"p95={row['p95']:.3f} p99={row['p99']:.3f}  -> {row['flows_per_s']:,.0f} flows/s")

print("\n[A2] Batched throughput, CPU 1 core (edge proxy)")
R["throughput_1c"]={}
for bs in [1,32,256,1024]:
    t=bench(1,bs,200); fps=bs/(t.mean()/1e3)
    R["throughput_1c"][bs]=float(fps)
    print(f"  batch={bs:4d}: {fps:>10,.0f} flows/s   (per-flow {t.mean()/bs:.4f} ms)")

if torch.cuda.is_available():
    torch.set_num_threads(DEFAULT_THREADS)
    xb=torch.tensor(X[:1024]).cuda(); mc=copy.deepcopy(main_model).cuda().eval()
    with torch.no_grad():
        for _ in range(10): mc(xb); torch.cuda.synchronize()
        s=time.perf_counter()
        for _ in range(200): mc(xb)
        torch.cuda.synchronize(); gt=(time.perf_counter()-s)/200
    print(f"\n[A3] GPU batch=1024: {1024/gt:,.0f} flows/s")
    R["gpu_throughput_1024"]=float(1024/gt)
torch.set_num_threads(DEFAULT_THREADS)

print("\n[B] Time-to-detection: the 27 features are flow-terminal aggregates")
I_DUR=present_features.index("Flow Duration")
dur_a=np.asarray(X_test_raw)[np.asarray(y_test)==1, I_DUR]/1e6   # seconds
dur_b=np.asarray(X_test_raw)[np.asarray(y_test)==0, I_DUR]/1e6
def q(x): return dict(median=float(np.median(x)),p95=float(np.percentile(x,95)),
                      p99=float(np.percentile(x,99)),max=float(x.max()))
R["attack_flow_dur_s"]=q(dur_a); R["benign_flow_dur_s"]=q(dur_b)
print(f"  attack-flow duration (s): median={np.median(dur_a):.2f}  p95={np.percentile(dur_a,95):.2f}  "
      f"p99={np.percentile(dur_a,99):.2f}  max={dur_a.max():.2f}")
print(f"  benign-flow duration (s): median={np.median(dur_b):.2f}  p95={np.percentile(dur_b,95):.2f}")
med_lat_ms=R["lat_b1_1c"]["median"]
print(f"\n  model latency per flow (1 core): {med_lat_ms:.3f} ms = {med_lat_ms/1000:.6f} s")
print(f"  median attack-flow lifetime    : {np.median(dur_a):.2f} s")
if np.median(dur_a)>0:
    print(f"  => flow lifetime is ~{np.median(dur_a)*1000/med_lat_ms:,.0f}x the model latency: the model is")
    print(f"     not the bottleneck; time-to-detection is bounded by flow completion / export timeout.")
json.dump(R,open(RES,"w"),indent=1)
print("\nsaved ->",RES)
