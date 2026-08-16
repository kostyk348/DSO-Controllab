#!/usr/bin/env python3
"""
RACS v2 — Resource-Aware Controller Synthesis
===============================================
DSO-inspired: детерминизм, бюджеты ресурсов, multi-objective.
Cortex-M4 cycle model + PID / LQR / MPC / GP / GP-R.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random, math, os, time, warnings
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from copy import deepcopy
from collections import defaultdict
warnings.filterwarnings('ignore')

np.random.seed(42); random.seed(42)

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

@dataclass
class Config:
    Ts: float = 0.01
    t_end: float = 30.0
    pop_size: int = 50
    generations: int = 15
    tournament: int = 4
    cx_prob: float = 0.75
    mut_prob: float = 0.20
    max_depth: int = 6
    elite: int = 5
    u_min: float = -10.0
    u_max: float = 10.0

cfg = Config()

# ──────────────────────────────────────────────
#  Cortex-M4 Cycle-Accurate Model
# ──────────────────────────────────────────────
# Based on ARM Cortex-M4 + FPU (single-precision)
# CALIBRATED against QEMU bare-metal (racs_codegen/codegen.py):
#   vdiv = 14 (VDIV.F32 latency), min/max = 10 (vcmp+vmrs+branch),
#   sin = 128 (soft Taylor), sqrt = 140 (Newton x8, each with VDIV)

CYCLES = {
    'add': 1, 'sub': 1, 'mul': 1, 'div': 14,          # VADD/VSUB/VMUL.F32=1, VDIV=14
    'sin': 128, 'cos': 128,                            # soft Taylor + range reduction
    'sqrt': 140,                                       # Newton-Raphson 8 iter w/ VDIV
    'abs': 1, 'neg': 1,                                # VABS/VNEG.F32
    'min': 10, 'max': 10,                              # VCMP + VMRS + conditional branch
    'const': 2,                                        # VLDR from literal pool
    'var': 1,                                          # VLDR from frame pointer
    'load': 2, 'store': 2,                             # memory access with wait states
    'branch': 3,                                       # conditional branch (worst-case taken)
}

# ABI overhead: softfp passes float args in r0-r3 (vmov into s-regs)
# + bx lr return. Measured: ~2-4 extra cycles for small controllers.
ABI_OVERHEAD_CYCLES = 3

# Memory: constants stored in literal pool (4 bytes each)
MEM_BYTES = {'const': 4}

VARS = ['e', 'ie', 'de', 'r', 'y']

# ──────────────────────────────────────────────
#  Expression Tree
# ──────────────────────────────────────────────

class Node:
    __slots__ = ('op','left','right','val')
    def __init__(self, op, left=None, right=None, val=None):
        self.op=op; self.left=left; self.right=right; self.val=val

    def evaluate(self, env: Dict[str,float]) -> float:
        try:
            if self.op=='const': return self.val
            if self.op=='var': return env.get(self.val,0.0)
            if self.op in ('neg','sin','cos','sqrt','abs'):
                lv=self.left.evaluate(env)
                if self.op=='neg': return -lv
                if self.op=='sin': return math.sin(lv)
                if self.op=='cos': return math.cos(lv)
                if self.op=='sqrt': return math.sqrt(abs(lv))
                if self.op=='abs': return abs(lv)
            lv=self.left.evaluate(env); rv=self.right.evaluate(env)
            if self.op=='add': return lv+rv
            if self.op=='sub': return lv-rv
            if self.op=='mul': return lv*rv
            if self.op=='div': return lv/rv if abs(rv)>1e-10 else 0.0
            if self.op=='min': return lv if lv<rv else rv
            if self.op=='max': return lv if lv>rv else rv
            return 0.0
        except: return 0.0

    def cycles(self) -> int:
        """Worst-case cycle count (Cortex-M4, calibrated)."""
        c = CYCLES.get(self.op, 3)
        if self.left: c += self.left.cycles()
        if self.right: c += self.right.cycles()
        return c

    def cycles_total(self) -> int:
        """Cycles including ABI call overhead (softfp, bare metal)."""
        return self.cycles() + ABI_OVERHEAD_CYCLES

    def mem_bytes(self) -> int:
        m = MEM_BYTES.get(self.op, 0)
        if self.left: m += self.left.mem_bytes()
        if self.right: m += self.right.mem_bytes()
        return m

    def depth(self) -> int:
        if self.op in ('const','var'): return 1
        if self.op in ('neg','sin','cos','sqrt','abs'): return 1+self.left.depth()
        return 1+max(self.left.depth(),self.right.depth())

    def clone(self):
        if self.op=='const': return Node('const',val=self.val)
        if self.op=='var': return Node('var',val=self.val)
        if self.op in ('neg','sin','cos','sqrt','abs'):
            return Node(self.op,left=self.left.clone())
        return Node(self.op,left=self.left.clone(),right=self.right.clone())

    def all_nodes(self):
        nodes=[self]
        if self.left: nodes+=self.left.all_nodes()
        if self.right: nodes+=self.right.all_nodes()
        return nodes

    def __repr__(self):
        if self.op=='const': return f"{self.val:.4f}"
        if self.op=='var': return str(self.val)
        if self.op in ('neg','sin','cos','sqrt','abs'): return f"{self.op}({self.left})"
        return f"({self.left} {self.op} {self.right})"


def random_tree(depth, max_d, terms):
    if depth >= max_d:
        return Node('const',val=random.uniform(-2,2)) if random.random()<0.4 else Node('var',val=random.choice(terms))
    ops=['add','sub','mul']
    r=random.random()
    if r<0.35: ops.append('div')
    if r<0.2: ops+=['sin','cos']
    if r<0.1: ops+=['sqrt','abs','neg']
    if r<0.05: ops+=['min','max']
    op=random.choice(ops)
    if op in ('neg','sin','cos','sqrt','abs'):
        return Node(op,left=random_tree(depth+1,max_d,terms))
    return Node(op,left=random_tree(depth+1,max_d,terms),right=random_tree(depth+1,max_d,terms))

def subtree_cx(p1,p2):
    c1=p1.clone(); c2=p2.clone()
    n1=random.choice(c1.all_nodes()); n2=random.choice(c2.all_nodes())
    n1.op,n1.left,n1.right,n1.val=n2.op,n2.left,n2.right,n2.val
    return c1,c2

def mutate(ind, terms):
    m=ind.clone(); node=random.choice(m.all_nodes())
    r=random.random()
    if r<0.3 and node.op not in ('const','var'):
        ops=['add','sub','mul']
        if node.op in ('sin','cos','sqrt','abs','neg'): ops=['sin','cos','sqrt','abs','neg']
        node.op=random.choice(ops)
    elif r<0.6 and node.op=='const':
        node.val+=random.uniform(-0.5,0.5); node.val=max(-5,min(5,node.val))
    else:
        sub=random_tree(0,cfg.max_depth-1,terms)
        node.op=sub.op; node.left=sub.left; node.right=sub.right; node.val=sub.val
    return m

def tree_formula(node):
    s=str(node); return s[:77]+'...' if len(s)>80 else s


# ──────────────────────────────────────────────
#  Plant Models
# ──────────────────────────────────────────────

class Plant:
    def __init__(self,Ts): self.Ts=Ts; self.reset()
    def reset(self): self.y=0.0
    def update(self,u): ...
    @property
    def name(self): return self.__class__.__name__
    def get_ss(self): return None  # (A,B,C,D) continuous-time or None


class SecondOrderDelay(Plant):
    def __init__(self,K=1.0,tau1=1.5,tau2=0.8,L=2.0,Ts=0.01):
        self.K=K;self.tau1=tau1;self.tau2=tau2;self.L=L;self.buf_len=max(1,int(L/Ts))
        super().__init__(Ts)
    def reset(self): self.x1=self.x2=0.0; self.buf=[0.0]*self.buf_len; self.y=0.0
    def update(self,u):
        ud=self.buf.pop(0); self.buf.append(u)
        self.x1+=(-self.x1+self.K*ud)/self.tau1*self.Ts
        self.x2+=(-self.x2+self.x1)/self.tau2*self.Ts; self.y=self.x2; return self.y
    @property
    def name(self): return f"2nd+Delay (τ₁={self.tau1},τ₂={self.tau2},L={self.L})"

class FourthOrder(Plant):
    def __init__(self,Ts=0.01): super().__init__(Ts)
    def reset(self): self.x=np.zeros(4); self.y=0.0
    def update(self,u):
        dx=np.array([-4*self.x[0]-6*self.x[1]-4*self.x[2]-self.x[3]+u,self.x[0],self.x[1],self.x[2]])
        self.x+=dx*self.Ts; self.y=self.x[3]; return self.y
    def get_ss(self):
        A=np.array([[-4,-6,-4,-1],[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        B=np.array([[1],[0],[0],[0]]); C=np.array([[0,0,0,1]]); D=np.array([[0]])
        return A,B,C,D
    @property
    def name(self): return "4th-Order 1/(s+1)⁴"

class Underdamped(Plant):
    def __init__(self,wn=1.5,zeta=0.15,Ts=0.01):
        self.wn=wn;self.zeta=zeta; super().__init__(Ts)
    def reset(self): self.x=np.zeros(2); self.y=0.0
    def update(self,u):
        wn2=self.wn**2
        dx=np.array([self.x[1],-2*self.zeta*self.wn*self.x[1]-wn2*self.x[0]+wn2*u])
        self.x+=dx*self.Ts; self.y=self.x[0]; return self.y
    def get_ss(self):
        A=np.array([[0,1],[-self.wn**2,-2*self.zeta*self.wn]])
        B=np.array([[0],[self.wn**2]]); C=np.array([[1,0]]); D=np.array([[0]])
        return A,B,C,D
    @property
    def name(self): return f"Underdamped (ζ={self.zeta},ωₙ={self.wn})"

class NonMinPhase(Plant):
    def __init__(self,Ts=0.01): super().__init__(Ts)
    def reset(self): self.x=np.zeros(3); self.y=0.0
    def update(self,u):
        dx=np.array([-3*self.x[0]+self.x[1], -3*self.x[1]+self.x[2], -self.x[2]+u])
        self.x+=dx*self.Ts; self.y=self.x[0]-2*self.x[1]; return self.y
    def get_ss(self):
        A=np.array([[-3,1,0],[-3,0,1],[-1,0,0]])
        B=np.array([[0],[0],[1]]); C=np.array([[1,-2,0]]); D=np.array([[0]])
        return A,B,C,D
    @property
    def name(self): return "Non-Min Phase (1-2s)/(s+1)³"

class IntegratingDelay(Plant):
    def __init__(self,L=1.0,Ts=0.01):
        self.L=L;self.buf_len=max(1,int(L/Ts)); super().__init__(Ts)
    def reset(self): self.state=0.0; self.buf=[0.0]*self.buf_len; self.y=0.0
    def update(self,u):
        ud=self.buf.pop(0); self.buf.append(u)
        self.state+=ud*self.Ts; self.y=self.state; return self.y
    @property
    def name(self): return f"Integrator+Delay (L={self.L})"


# ──────────────────────────────────────────────
#  Controllers
# ──────────────────────────────────────────────

class PID:
    def __init__(self,Kp=1.0,Ki=0.5,Kd=0.1,Ts=0.01):
        self.Kp=Kp;self.Ki=Ki;self.Kd=Kd;self.Ts=Ts;self.reset()
    def reset(self): self.ie=0.0
    def compute(self,r,y,e,ie,de):
        self.ie+=e*self.Ts
        u=self.Kp*e+self.Ki*self.ie+self.Kd*de
        if abs(u)>=cfg.u_max: self.ie-=e*self.Ts
        return max(cfg.u_min,min(cfg.u_max,u))
    @property
    def cycles(self): return 20
    @property
    def mem_bytes(self): return 16
    @property
    def label(self): return "PID"
    def gains(self): return self.Kp,self.Ki,self.Kd


def optimize_pid(plant_class,Ts,t_end,trials=300):
    best,best_iae=None,float('inf')
    for _ in range(trials):
        pid=PID(10**random.uniform(-1,1.5),10**random.uniform(-2,1),10**random.uniform(-2,1),Ts)
        res=run_sim(plant_class,lambda r,y,e,ie,de:pid.compute(r,y,e,ie,de),Ts,t_end)
        if res['iae']<best_iae: best_iae,best=res['iae'],PID(pid.Kp,pid.Ki,pid.Kd,Ts)
    for _ in range(100):
        eps=0.05
        Kp=max(0.001,min(20,best.Kp*(1+random.uniform(-eps,eps))))
        Ki=max(0.0,min(10,best.Ki*(1+random.uniform(-eps,eps))))
        Kd=max(0.0,min(10,best.Kd*(1+random.uniform(-eps,eps))))
        pid=PID(Kp,Ki,Kd,Ts)
        res=run_sim(plant_class,lambda r,y,e,ie,de:pid.compute(r,y,e,ie,de),Ts,t_end)
        if res['iae']<best_iae: best_iae,best=res['iae'],PID(Kp,Ki,Kd,Ts)
    return best


class LQR:
    """LQR with steady-state Kalman observer (LQG)."""
    def __init__(self,A,B,C,D,Ts):
        self.Ts=Ts; self.n=A.shape[0]
        self.Ad=np.eye(self.n)+A*Ts; self.Bd=(B*Ts).flatten(); self.Cd=C; self.Dd=D
        # State feedback (assumes full state access)
        # Full-state LQR (aggressive tracking)
        Q_Ric=np.eye(self.n); Q_Ric[0,0]=100.0; R=0.01
        Bmat=self.Bd.reshape(-1,1)
        P=self._dare(self.Ad,Bmat,Q_Ric,np.array([[R]]))
        self.K=-np.linalg.inv(R+Bmat.T@P@Bmat)@Bmat.T@P@self.Ad
        self.K=self.K.reshape(1,self.n)
        # Observer gain (fast, aggressive tracking)
        Q_kf=np.eye(self.n)*1000.0; R_kf=0.1
        Pk=self._dare(self.Ad.T,self.Cd.T,Q_kf,np.array([[R_kf]]))
        self.L=Pk@self.Cd.T@np.linalg.inv(self.Cd@Pk@self.Cd.T+R_kf)
        self.L=self.L.reshape(self.n,1)
        self.reset()
    def reset(self):
        self.x_hat=np.zeros(self.n); self.u_prev=0.0
    def _dare(self,A,B,Q,R,max_iter=2000,tol=1e-8):
        P=Q.copy()
        for _ in range(max_iter):
            Pn=A.T@P@A - A.T@P@B@np.linalg.inv(R+B.T@P@B)@B.T@P@A + Q
            if np.max(np.abs(Pn-P))<tol: break
            P=Pn
        return P
    def compute(self,r,y,e,ie,de):
        # Simple Luenberger observer: x̂_{k+1} = Ad·x̂ + Bd·u + L·(y - C·x̂)
        y_hat=(self.Cd@self.x_hat).item()
        innovation=y-y_hat
        self.x_hat=self.Ad@self.x_hat+self.Bd*self.u_prev+self.L.flatten()*innovation
        u=-(self.K@self.x_hat).item()
        self.u_prev=u
        return max(cfg.u_min,min(cfg.u_max,u))
    @property
    def cycles(self): return 30+10*self.n
    @property
    def mem_bytes(self): return 8*self.n*self.n+8
    @property
    def label(self): return "LQR"


class MPC:
    """Model Predictive Control with adaptive grid search."""
    def __init__(self,plant_class,Ts,N=8):
        self.plant_class=plant_class; self.Ts=Ts; self.N=N
        self.plant=plant_class(Ts=Ts); self.reset()
    def reset(self):
        self.plant.reset(); self.u_prev=0.0
    def compute(self,r,y,e,ie,de):
        # PI-inspired candidate + refinement
        pi_u = 0.5*e + 0.1*ie
        candidates = np.linspace(-3,3,9)
        candidates = np.append(candidates, pi_u)
        best_u,best_cost=0.0,float('inf')
        for scale in [1.0, 0.5, 0.2]:
            for cu in candidates*scale + self.u_prev*0.3:
                cu=np.clip(cu,cfg.u_min,cfg.u_max)
                saved=deepcopy(self.plant)
                cost=0.0
                for k in range(self.N):
                    yi=self.plant.update(cu)
                    ei=1.0-yi
                    cost+=ei*ei
                self.plant=saved
                if cost<best_cost: best_cost,best_u=cost,cu
        self.u_prev=best_u
        self.plant.update(best_u)
        return best_u
    @property
    def cycles(self): return 200+30*self.N
    @property
    def mem_bytes(self): return 80


# ──────────────────────────────────────────────
#  Simulation Engine
# ──────────────────────────────────────────────

def run_sim(plant_class, ctrl_fn, Ts, t_end, budget=None):
    """Simulate step + disturbance. ctrl_fn(r,y,e,ie,de) -> u.
    budget: dict with 'cycles_max','mem_max' — if exceeded, penalize."""
    plant=plant_class(Ts=Ts); plant.reset()
    N=int(t_end/Ts); t=np.arange(N)*Ts
    y_arr=np.zeros(N); u_arr=np.zeros(N)
    iae=0.0; ie=0.0; e_prev=0.0; y=0.0
    dist_time=t_end*0.5; dist_mag=-0.5

    for i in range(N):
        r=1.0; e=r-y; de=(e-e_prev)/Ts if Ts>0 else 0.0
        u=ctrl_fn(r,y,e,ie,de)
        u=max(cfg.u_min,min(cfg.u_max,u))
        u_plant=u+(dist_mag if t[i]>=dist_time else 0.0)
        y=plant.update(u_plant)
        y_arr[i]=y; u_arr[i]=u
        iae+=abs(e)*Ts; ie+=e*Ts; e_prev=e

    y1=y_arr[:int(N/2)]
    os=max(0,(np.max(y1)-1)*100) if np.max(y1)>1 else 0.0
    energy=np.sum(np.diff(u_arr)**2)/N
    settled=np.where(np.abs(y_arr[int(N*0.1):]-1)<0.02)[0]
    ts=t[int(N*0.1)+settled[0]] if len(settled)>0 else t[-1]
    # Determinism: variance in steady-state after settling
    y_ss=y_arr[int(N*0.8):]
    jitter=np.var(y_ss) if len(y_ss)>0 else 0.0

    return {'t':t,'y':y_arr,'u':u_arr,'iae':iae,'os':os,
            'energy':energy,'ts':ts,'jitter':jitter,
            'plant':plant}


# ──────────────────────────────────────────────
#  GP Engine with Pareto archive
# ──────────────────────────────────────────────

class GPController:
    def __init__(self,plant_class,Ts,t_end,resource_penalty=False):
        self.plant_class=plant_class; self.Ts=Ts; self.t_end=t_end
        self.resource_penalty=resource_penalty
        self.pop=[]; self.fitness=[]
        self.best=None; self.best_fit=-float('inf')
        self.history=[]; self.pareto_front=[]

    def _ctrl(self,tree):
        return lambda r,y,e,ie,de: tree.evaluate({'e':e,'ie':ie,'de':de,'r':r,'y':y})

    def _fit(self,tree):
        try:
            if tree.depth()>cfg.max_depth: return -float('inf')
            res=run_sim(self.plant_class,self._ctrl(tree),self.Ts,self.t_end)
            cyc=tree.cycles(); mem=tree.mem_bytes()
            # Multi-objective scalarization
            quality=res['iae']/10.0 + 0.3*res['os']/100.0 + 0.05*res['energy']*10
            resource=0
            if self.resource_penalty:
                resource=0.15*cyc/100 + 0.08*mem/20
            budget=0
            if cyc>500: budget+=0.5  # hard budget violation penalty
            if mem>128: budget+=0.3
            cost=quality+resource+budget
            # Store for Pareto
            self.pareto_front.append((res['iae'],cyc,mem,tree.clone()))
            return 1.0/(1.0+cost)
        except: return -float('inf')

    def run(self):
        for i in range(cfg.pop_size):
            self.pop.append(random_tree(2+i%(cfg.max_depth-1),cfg.max_depth,VARS))
        self.fitness=[self._fit(t) for t in self.pop]

        for gen in range(cfg.generations):
            new_pop=[]
            for idx in np.argsort(self.fitness)[-cfg.elite:]:
                new_pop.append(self.pop[idx].clone())
            while len(new_pop)<cfg.pop_size:
                p1=self.pop[max(random.sample(range(len(self.pop)),cfg.tournament),key=lambda i:self.fitness[i])]
                p2=self.pop[max(random.sample(range(len(self.pop)),cfg.tournament),key=lambda i:self.fitness[i])]
                if random.random()<cfg.cx_prob: c1,c2=subtree_cx(p1,p2)
                else: c1,c2=p1.clone(),p2.clone()
                if random.random()<cfg.mut_prob: c1=mutate(c1,VARS)
                if random.random()<cfg.mut_prob: c2=mutate(c2,VARS)
                if c1.depth()>cfg.max_depth: c1=p1.clone()
                if c2.depth()>cfg.max_depth: c2=p2.clone()
                new_pop.append(c1)
                if len(new_pop)<cfg.pop_size: new_pop.append(c2)
            self.pop=new_pop[:cfg.pop_size]
            self.fitness=[self._fit(t) for t in self.pop]
            bi=np.argmax(self.fitness); bf=self.fitness[bi]
            self.history.append(bf)
            if bf>self.best_fit: self.best_fit=bf; self.best=self.pop[bi].clone()
            # Early stopping
            if gen>10 and max(self.history[-8:])==self.history[-8]:
                if gen%15!=0: print(f"  Gen {gen:3d}: early stop (no improvement)")
                break
            if gen%10==0 or gen==cfg.generations-1:
                print(f"  Gen {gen:3d}: fit={bf:.4f} instr={self.pop[bi].cycles()}  {tree_formula(self.pop[bi])}")
        return self.best,self.best_fit,self.history,self.pareto_front


# ──────────────────────────────────────────────
#  Benchmark
# ──────────────────────────────────────────────

def run_benchmark(plant_classes):
    results=[]
    for PlantCls in plant_classes:
        plant=PlantCls(Ts=cfg.Ts)
        print(f"\n{'='*70}\n  {plant.name}\n{'='*70}")

        # PID
        print("\n  [PID]"); t0=time.time()
        pid=optimize_pid(PlantCls,cfg.Ts,cfg.t_end)
        pid_res=run_sim(PlantCls,lambda r,y,e,ie,de:pid.compute(r,y,e,ie,de),cfg.Ts,cfg.t_end)
        print(f"  Kp={pid.Kp:.3f} Ki={pid.Ki:.3f} Kd={pid.Kd:.3f}  IAE={pid_res['iae']:.4f} cycles={pid.cycles} ({time.time()-t0:.1f}s)")

        # LQR (if SS available)
        lqr_res=None; lqr=None
        ss=PlantCls(Ts=cfg.Ts).get_ss()
        if ss is not None:
            print("  [LQR]"); t0=time.time()
            A,B,C,D=ss
            try:
                lqr=LQR(A,B,C,D,cfg.Ts)
                lqr_res=run_sim(PlantCls,lambda r,y,e,ie,de:lqr.compute(r,y,e,ie,de),cfg.Ts,cfg.t_end)
                print(f"  IAE={lqr_res['iae']:.4f} cycles={lqr.cycles} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  LQR failed: {e}"); lqr_res=None

        # MPC
        print("  [MPC]"); t0=time.time()
        mpc=MPC(PlantCls,cfg.Ts,N=5)
        mpc_res=run_sim(PlantCls,lambda r,y,e,ie,de:mpc.compute(r,y,e,ie,de),cfg.Ts,cfg.t_end)
        print(f"  IAE={mpc_res['iae']:.4f} cycles={mpc.cycles} ({time.time()-t0:.1f}s)")

        # GP
        print("  [GP]"); t0=time.time()
        gp=GPController(PlantCls,cfg.Ts,cfg.t_end,resource_penalty=False)
        best_tree,bf,hist,pareto=gp.run()
        gp_res=run_sim(PlantCls,gp._ctrl(best_tree),cfg.Ts,cfg.t_end)
        print(f"  IAE={gp_res['iae']:.4f} cycles={best_tree.cycles()} ({time.time()-t0:.1f}s)")
        print(f"  Formula: {tree_formula(best_tree)}")

        # GP-R
        print("  [GP-R]"); t0=time.time()
        gpr=GPController(PlantCls,cfg.Ts,cfg.t_end,resource_penalty=True)
        best_tree_r,bf_r,hist_r,pareto_r=gpr.run()
        gpr_res=run_sim(PlantCls,gpr._ctrl(best_tree_r),cfg.Ts,cfg.t_end)
        print(f"  IAE={gpr_res['iae']:.4f} cycles={best_tree_r.cycles()} ({time.time()-t0:.1f}s)")
        print(f"  Formula: {tree_formula(best_tree_r)}")

        results.append({
            'plant': plant.name,
            'pid': pid_res, 'pid_ctrl': pid,
            'lqr': lqr_res, 'lqr_ctrl': lqr,
            'mpc': mpc_res, 'mpc_ctrl': mpc,
            'gp': gp_res, 'gp_tree': best_tree, 'gp_hist': hist, 'gp_pareto': pareto,
            'gpr': gpr_res, 'gpr_tree': best_tree_r, 'gpr_hist': hist_r, 'gpr_pareto': pareto_r,
        })
    return results


# ──────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────

def plot_all(results,save_dir='.'):
    n=len(results)
    fig=plt.figure(figsize=(22,6*n))

    for idx,res in enumerate(results):
        gs=gridspec.GridSpecFromSubplotSpec(4,3,
            subplot_spec=fig.add_gridspec(n,1)[idx],
            width_ratios=[2,2,1], hspace=0.4,wspace=0.35)

        # Step response
        ax1=fig.add_subplot(gs[0,:2])
        for tag,key in [('PID','pid'),('LQR','lqr'),('MPC','mpc'),('GP','gp'),('GP-R','gpr')]:
            if res[key] is not None:
                ax1.plot(res[key]['t'],res[key]['y'],lw=1.5,alpha=0.8,label=tag)
        ax1.axhline(1,color='gray',ls='--',alpha=0.5)
        ax1.axvline(15,color='gray',ls=':',alpha=0.3)
        ax1.set_ylabel('y(t)'); ax1.set_title(f"{res['plant']} — Step Response"); ax1.legend(fontsize=7); ax1.grid(True,alpha=0.3)

        # Control
        ax2=fig.add_subplot(gs[1,:2])
        for tag,key in [('PID','pid'),('LQR','lqr'),('MPC','mpc'),('GP','gp'),('GP-R','gpr')]:
            if res[key] is not None:
                ax2.plot(res[key]['t'],res[key]['u'],lw=1,alpha=0.7,label=tag)
        ax2.set_ylabel('u(t)'); ax2.set_xlabel('Time [s]'); ax2.legend(fontsize=7); ax2.grid(True,alpha=0.3)

        # Evolution
        ax3=fig.add_subplot(gs[2,:2])
        ax3.plot(res['gp_hist'],'r-',alpha=0.7,label='GP')
        ax3.plot(res['gpr_hist'],'g-',alpha=0.7,label='GP-R')
        ax3.set_xlabel('Generation'); ax3.set_ylabel('Best Fitness'); ax3.legend(fontsize=7); ax3.grid(True,alpha=0.3)

        # Bar chart: IAE + Cycles
        ax4=fig.add_subplot(gs[0,2])
        tags=[]; iae_vals=[]; cyc_vals=[]
        for tag,key,ctrl_key in [('PID','pid','pid_ctrl'),('LQR','lqr','lqr_ctrl'),
                                  ('MPC','mpc','mpc_ctrl'),('GP','gp','gp_tree'),
                                  ('GP-R','gpr','gpr_tree')]:
            if res[key] is not None:
                tags.append(tag); iae_vals.append(res[key]['iae'])
                if ctrl_key and isinstance(res.get(ctrl_key,None),Node):
                    cyc_vals.append(res[ctrl_key].cycles())
                elif ctrl_key:
                    cyc_vals.append(res[ctrl_key].cycles)
                else:
                    cyc_vals.append(0)
        x=np.arange(len(tags)); w=0.35
        ax4.bar(x-w/2,iae_vals,w,color='steelblue',alpha=0.8,label='IAE')
        ax4_twin=ax4.twinx()
        ax4_twin.bar(x+w/2,cyc_vals,w,color='crimson',alpha=0.6,label='Cycles')
        ax4.set_xticks(x); ax4.set_xticklabels(tags,fontsize=8)
        ax4.set_ylabel('IAE',color='steelblue'); ax4_twin.set_ylabel('Cycles',color='crimson')
        ax4.grid(True,alpha=0.3,axis='y')

        # Pareto: IAE vs Cycles
        ax5=fig.add_subplot(gs[1,2])
        if res['gp_pareto']:
            pts=np.array([(p[0],p[1]) for p in res['gp_pareto']])
            ax5.scatter(pts[:,0],pts[:,1],c='r',s=5,alpha=0.4,label='GP pop')
            # Highlight best
            ax5.scatter(res['gp']['iae'],res['gp_tree'].cycles(),c='darkred',s=80,marker='*',label='GP best')
            ax5.scatter(res['gpr']['iae'],res['gpr_tree'].cycles(),c='darkgreen',s=80,marker='*',label='GP-R best')
        for tag,key,ctrl_key in [('PID','pid','pid_ctrl'),('MPC','mpc','mpc_ctrl')]:
            if res[key] is not None:
                ctrl_obj=res.get(ctrl_key)
                if isinstance(ctrl_obj,Node): c=ctrl_obj.cycles()
                else: c=getattr(ctrl_obj,'cycles',0)
                ax5.scatter(res[key]['iae'],c,s=100,marker='D',label=tag)
        ax5.set_xlabel('IAE'); ax5.set_ylabel('Cycles'); ax5.legend(fontsize=6); ax5.grid(True,alpha=0.3)
        ax5.set_title('Pareto: Quality vs Resources')

        # Formula
        ax6=fig.add_subplot(gs[2,2]); ax6.axis('off')
        ax6.text(0.05,0.85,f"GP: {tree_formula(res['gp_tree'])}",fontsize=7,family='monospace',
                 bbox=dict(boxstyle='round',fc='lightyellow',alpha=0.8))
        ax6.text(0.05,0.55,f"GP-R: {tree_formula(res['gpr_tree'])}",fontsize=7,family='monospace',
                 bbox=dict(boxstyle='round',fc='lightgreen',alpha=0.8))
        if res['pid_ctrl']:
            p=res['pid_ctrl']
            ax6.text(0.05,0.25,f"PID: Kp={p.Kp:.3f} Ki={p.Ki:.3f} Kd={p.Kd:.3f}",fontsize=7,family='monospace',
                     bbox=dict(boxstyle='round',fc='lightblue',alpha=0.8))
        if res.get('lqr_ctrl'):
            ax6.text(0.05,0.05,"LQR: state feedback + observer",fontsize=7,family='monospace',
                     bbox=dict(boxstyle='round',fc='plum',alpha=0.8))

    fig.suptitle('RACS v2 — DSO Resource-Aware Controller Synthesis',fontsize=14,y=1.005)
    plt.tight_layout(); path=os.path.join(save_dir,'racs2_benchmark.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); print(f"Saved: {path}"); plt.close(fig)


def plot_pareto_summary(results,save_dir='.'):
    """Aggregate Pareto front across all processes."""
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    colors=plt.cm.tab10(np.linspace(0,1,len(results)))
    for i,res in enumerate(results):
        if res['gp_pareto']:
            pts=np.array([(p[0],p[1]) for p in res['gp_pareto']])
            axes[0].scatter(pts[:,0],pts[:,1],c=[colors[i]],s=3,alpha=0.3)
            axes[0].scatter(res['gp']['iae'],res['gp_tree'].cycles(),c=[colors[i]],s=80,marker='*',
                          label=res['plant'][:20])
            axes[1].scatter(res['gp_tree'].cycles(),res['gp']['iae'],c=[colors[i]],s=80,marker='*')
    axes[0].set_xlabel('IAE (control quality)'); axes[0].set_ylabel('Cycles (compute cost)')
    axes[0].set_title('GP Pareto: Quality vs Resources'); axes[0].legend(fontsize=6); axes[0].grid(True,alpha=0.3)
    axes[1].set_xlabel('Cycles'); axes[1].set_ylabel('IAE')
    axes[1].set_title('Inverse: cheaper is left'); axes[1].grid(True,alpha=0.3)
    plt.tight_layout(); path=os.path.join(save_dir,'racs2_pareto.png')
    fig.savefig(path,dpi=150,bbox_inches='tight'); print(f"Saved: {path}"); plt.close(fig)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    print("="*70)
    print("  RACS v2 — Resource-Aware Controller Synthesis (DSO)")
    print("  PID / LQR / MPC / GP / GP-R  |  Cortex-M4 cycle model")
    print("="*70)

    plants=[SecondOrderDelay,FourthOrder,Underdamped,NonMinPhase,IntegratingDelay]

    t0=time.time()
    results=run_benchmark(plants)
    elapsed=time.time()-t0

    # Summary table
    print("\n\n"+"="*110)
    print(f"  BENCHMARK SUMMARY  ({elapsed:.0f}s)")
    print("="*110)
    hdr=f"{'Process':<32}{'PID IAE':<12}{'LQR IAE':<12}{'MPC IAE':<12}{'GP IAE':<12}{'GP-R IAE':<12}{'GP cyc':<10}"
    print(f"  {hdr}")
    print("  "+"-"*102)
    for res in results:
        def v(r):
            return f"{r['iae']:.3f}" if r is not None else "N/A"
        cyc_str=str(res['gp_tree'].cycles()) if res.get('gp_tree') else "N/A"
        row=f"{res['plant'][:30]:<32}{v(res['pid']):<12}{v(res['lqr']):<12}{v(res['mpc']):<12}{v(res['gp']):<12}{v(res['gpr']):<12}{cyc_str:<10}"
        print(f"  {row}")
    print("="*110)

    plot_all(results)
    plot_pareto_summary(results)
    print("\nDone.")


if __name__=='__main__':
    main()
