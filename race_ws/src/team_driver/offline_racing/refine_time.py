#!/usr/bin/env python3
"""Original reduced-order minimum-profile-time refinement, not full dynamic optimal control.
Optimises smooth lateral offsets under the original corridor and steering bounds.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from generate_raceline import TrackMap,geometry,velocity,dense_segments

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--map',required=True);a.add_argument('--centreline',required=True)
    a.add_argument('--seed',required=True);a.add_argument('--output',required=True)
    a.add_argument('--knots',type=int,default=40);a.add_argument('--iterations',type=int,default=80)
    args=a.parse_args()
    p=np.loadtxt(args.centreline,delimiter=',')[:,:2]
    seed=np.loadtxt(args.seed,delimiter=',')[:,:2]
    if p.shape!=seed.shape:raise ValueError('Seed must retain centreline correspondence')
    m=TrackMap(args.map)
    tangent=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    normals=np.column_stack((-tangent[:,1],tangent[:,0]))
    alpha=np.sum((seed-p)*normals,axis=1)
    if np.max(np.linalg.norm(p+alpha[:,None]*normals-seed,axis=1))>1e-5:
        raise ValueError('Seed is not in the normal-offset corridor')
    # The same contiguous safe intervals used by the curvature generator.
    offsets=np.arange(0,.651,.01);bounds=[]
    for sign in [-1,1]:
        safe=m.clearance(p[:,None,:]+sign*offsets[None,:,None]*normals[:,None,:])>=.42
        safe[:,0]=True
        bounds.append(np.array([offsets[max(0,int(np.flatnonzero(~r)[0])-1)] if np.any(~r) else offsets[-1] for r in safe]))
    low,high=-bounds[0],bounds[1]
    knots=args.knots
    basis=CubicSpline(np.linspace(0,len(p),knots+1),np.vstack((np.eye(knots),np.eye(knots)[0])),bc_type='periodic')(np.arange(len(p)))
    def path(beta):return p+(alpha+basis@beta)[:,None]*normals
    def time_for(q):
        ds,k=geometry(q);v=velocity(q)
        return float(np.sum(2*ds/(v+np.roll(v,-1))))
    def objective(beta):
        q=path(beta);_,k=geometry(q)
        return time_for(q)+.015*np.mean(k*k)+.01*np.mean(beta*beta)
    def constraints(beta):
        aa=alpha+basis@beta;q=path(beta);ds,k=geometry(q)
        return np.r_[aa-low,high-aa,1.25-np.abs(k),ds-.07]
    opt=minimize(objective,np.zeros(knots),method='SLSQP',bounds=[(-.35,.35)]*knots,
                 constraints={'type':'ineq','fun':constraints},options={'maxiter':args.iterations,'ftol':1e-7})
    choices=[(time_for(seed),seed,0.)]
    for scale in np.linspace(1,0,21):
        beta=scale*opt.x;q=path(beta)
        if np.min(constraints(beta))< -1e-6:continue
        if m.clearance(dense_segments(q)).min()<.40:continue
        if any(m.clearance(dense_segments(p+t*(q-p),.04)).min()<.40 for t in np.linspace(0,1,31)):continue
        choices.append((time_for(q),q,float(scale)))
    score,q,fraction=min(choices,key=lambda x:x[0])
    np.savetxt(args.output,np.column_stack((q,velocity(q))),delimiter=',',fmt='%.8f',header='x_m,y_m,v_mps; time-profile refinement; NOT official simulator validated')
    report={'method':'reduced-order constrained profile-time refinement','optimizer_success':bool(opt.success),'message':str(opt.message),
            'iterations':int(opt.nit),'selected_fraction':fraction,'seed_ideal_seconds':time_for(seed),'output_ideal_seconds':score,
            'min_reference_clearance_m':float(m.clearance(dense_segments(q)).min()),'length_m':float(geometry(q)[0].sum()),
            'peak_curvature':float(abs(geometry(q)[1]).max()),'arguments':vars(args),
            'warning':'Model proxy only; excludes tyre slip, feedback tracking and LiDAR interventions. Feasibility post-check required even on solver success.'}
    Path(args.output+'.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
