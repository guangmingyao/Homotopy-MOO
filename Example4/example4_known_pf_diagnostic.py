import numpy as np
import pandas as pd
import example4_homotopy_dtlz_v4 as B

# Example 4 diagnostic: two 3-objective problems with known PFs.
# Put this script beside example4_homotopy_dtlz_v4.py.

N, M, P = 2, 3, 3
B.D, B.M, B.P = N, M, P

def z_from_x(x):
    x1, x2 = x
    return np.array([x1, x2, 1-x1-x2], float)

def inequalities(x):
    x1, x2 = x
    return np.array([-x1, -x2, x1+x2-1], float)

def grad_inequalities(x):
    return np.array([[-1.,0.],[0.,-1.],[1.,1.]])

def hess_inequalities(x):
    return [np.zeros((2,2)) for _ in range(3)]

def objectives(x, problem):
    z = z_from_x(x)
    if problem == "convex":
        return z**2
    if problem == "concave":
        return -(z**2)
    raise ValueError(problem)

def grad_objectives(x, problem):
    x1, x2 = x
    G = np.array([[2*x1,0.],[0.,2*x2],
                  [-2*(1-x1-x2),-2*(1-x1-x2)]])
    return G if problem == "convex" else -G

def hess_objectives(x, problem):
    H = [np.array([[2.,0.],[0.,0.]]),
         np.array([[0.,0.],[0.,2.]]),
         np.array([[2.,2.],[2.,2.]])]
    return H if problem == "convex" else [-A for A in H]

B.objectives = objectives
B.grad_objectives = grad_objectives
B.hess_objectives = hess_objectives
B.inequalities = inequalities
B.grad_inequalities = grad_inequalities
B.hess_inequalities = hess_inequalities

def exact_x(lam):
    q = 1/np.asarray(lam,float)
    z = q/q.sum()
    return z[:2]

def weights(level=8):
    out=[]
    for i in range(1,level):
        for j in range(1,level-i):
            k=level-i-j
            if k>=1:
                out.append(np.array([i,j,k],float)/level)
    return out

W=weights(8) # 21 strictly positive weights
starts={
    "center":np.array([1/3,1/3]),
    "left":np.array([.15,.425]),
    "right":np.array([.425,.15]),
    "upper":np.array([.425,.425])
}

rows=[]
for problem in ["convex","concave"]:
    print("\n"+"="*68+"\n"+problem.upper()+"\n"+"="*68)
    for iw,lam0 in enumerate(W,1):
        xs=exact_x(lam0)
        Fs=objectives(xs,problem)
        for sname,x0 in starts.items():
            om,success,k,acc,rej,step,res=B.run_single_path(
                problem,x0,lam0,u0_scale=.05,
                step0=.06,step_min=1e-5,step_max=.10,
                eps_mu=1e-4,max_steps=900)
            x,lam,u,mu=B.unpack_omega(om)
            F=objectives(x,problem)
            feasible=np.max(inequalities(x))<=1e-7
            z=z_from_x(x)
            rows.append(dict(
                problem=problem,weight_id=iw,
                lambda1=lam0[0],lambda2=lam0[1],lambda3=lam0[2],
                start=sname,success=success,feasible=feasible,mu=mu,
                x1=x[0],x2=x[1],f1=F[0],f2=F[1],f3=F[2],
                exact_x1=xs[0],exact_x2=xs[1],
                x_error=np.linalg.norm(x-xs),
                F_error=np.linalg.norm(F-Fs),
                lambda_move=np.linalg.norm(lam-lam0),
                min_z=np.min(z),boundary=np.min(z)<1e-4,
                accepted=acc,rejected=rej))
    sub=pd.DataFrame(rows)
    sub=sub[sub.problem==problem]
    good=sub.success & sub.feasible
    near=good & (sub.F_error<1e-3)
    print("paths                 :",len(sub))
    print("successful + feasible :",int(good.sum()))
    print("near analytic KKT PF  :",int(near.sum()))
    print("boundary endpoints    :",int((good & sub.boundary).sum()))
    if good.any():
        print("median F error         :",f"{sub.loc[good,'F_error'].median():.6e}")
        print("max F error            :",f"{sub.loc[good,'F_error'].max():.6e}")
        print("max lambda movement    :",f"{sub.loc[good,'lambda_move'].max():.6e}")

df=pd.DataFrame(rows)
df.to_csv("example4_known_pf_diagnostic.csv",index=False)
print("\nSUMMARY BY PROBLEM / START")
tmp=df.assign(good=df.success&df.feasible,
              near=df.success&df.feasible&(df.F_error<1e-3))
print(tmp.groupby(["problem","start"]).agg(
    paths=("weight_id","count"),good=("good","sum"),
    near_exact=("near","sum"),median_F_error=("F_error","median"),
    max_F_error=("F_error","max"),boundary=("boundary","sum")
).to_string())
print("\nSaved: example4_known_pf_diagnostic.csv")
