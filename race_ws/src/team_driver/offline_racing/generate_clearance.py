#!/usr/bin/env python3
"""Original conservative normal-offset curvature optimiser; offline dependencies only.
Research inspiration: TUMFTM global_racetrajectory_optimization and BDEvan5
f1tenth_benchmarks. No code copied. See ENGINEERING.md for limitations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import yaml
from scipy.ndimage import label, distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def geometry(p):
    a = p - np.roll(p, 1, axis=0)
    b = np.roll(p, -1, axis=0) - p
    ds = np.linalg.norm(b, axis=1)
    k = 2 * (a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]) / np.maximum(
        np.linalg.norm(a, axis=1)*ds*np.linalg.norm(a+b, axis=1), 1e-10)
    return ds, k


def velocity(p, vmax=5.8, ay=5., accel=4., brake=3.5):
    ds, k = geometry(p)
    # No forced minimum: it must not override curvature feasibility.
    v = np.minimum(vmax, np.sqrt(ay/np.maximum(np.abs(k), 1e-6)))
    # Periodic envelope: iterate across the seam until both constraints converge.
    for _ in range(len(p)+1):
        old = v.copy()
        for i in range(len(p)-1, -1, -1):
            j = (i+1) % len(p)
            v[i] = min(v[i], np.sqrt(v[j]**2 + 2*brake*ds[i]))
        for i in range(len(p)):
            j = (i+1) % len(p)
            v[j] = min(v[j], np.sqrt(v[i]**2 + 2*accel*ds[i]))
        if np.max(np.abs(old-v)) < 1e-9:
            break
    return v


class TrackMap:
    def __init__(self, filename, seal_gap=.55):
        meta = yaml.safe_load(Path(filename).read_text())
        self.res = float(meta['resolution'])
        self.origin = np.array(meta['origin'][:2], dtype=float)
        if abs(meta['origin'][2]) > 1e-9:
            raise ValueError('Only zero-yaw map origins supported')
        self.image = np.array(Image.open(Path(filename).parent/meta['image']).convert('L'))
        occupied = self.image < 128  # simulator convention, NOT map_server threshold
        components, n = label(occupied)
        groups = [np.argwhere(components == i) for i in range(1, n+1)]
        canvas = Image.fromarray((occupied*255).astype('uint8'))
        draw = ImageDraw.Draw(canvas)
        self.links = []
        # Join all physically adjacent black components across short gaps. This
        # conservatively seals cone chains, without hand-selecting track locations.
        trees = [cKDTree(g) for g in groups]
        for i in range(n):
            for j in range(i+1, n):
                d, ix = trees[i].query(groups[j])
                m = int(np.argmin(d))
                if d[m]*self.res <= seal_gap:
                    a, b = groups[i][ix[m]], groups[j][m]
                    draw.line((int(a[1]),int(a[0]),int(b[1]),int(b[0])), fill=255, width=2)
                    self.links.append([a.tolist(), b.tolist()])
        self.occupied = np.array(canvas) > 0
        # Outside map is occupied; subtract pixel diagonal to bound raster error.
        padded = np.pad(~self.occupied, 1, constant_values=False)
        self.distance = distance_transform_edt(padded)[1:-1,1:-1]*self.res - self.res*np.sqrt(2)

    def clearance(self, p):
        p = np.asarray(p)
        x = (p[...,0]-self.origin[0])/self.res
        y = self.image.shape[0]-1-(p[...,1]-self.origin[1])/self.res
        return map_coordinates(self.distance, [y.ravel(), x.ravel()], order=1,
                               mode='constant', cval=-1.).reshape(x.shape)


def dense_segments(p, spacing=.02):
    q = np.roll(p, -1, axis=0)
    parts = []
    for a,b in zip(p,q):
        count = max(2, int(np.ceil(np.linalg.norm(b-a)/spacing))+1)
        parts.append(a+(b-a)*np.linspace(0,1,count)[:,None])
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map', required=True)
    ap.add_argument('--centreline', required=True)
    ap.add_argument('--output', default='icra26_raceline.csv')
    ap.add_argument('--clearance', type=float, default=.40,
                    help='reference-point radius including body and tracking margin')
    ap.add_argument('--max-offset', type=float, default=.65)
    ap.add_argument('--seal-gap', type=float, default=.55)
    ap.add_argument('--max-speed', type=float, default=5.8)
    ap.add_argument('--lateral-accel', type=float, default=5.)
    ap.add_argument('--accel', type=float, default=4.)
    ap.add_argument('--brake', type=float, default=3.5)
    args = ap.parse_args()
    if min(args.clearance,args.max_offset,args.max_speed,args.lateral_accel,args.accel,args.brake)<=0:
        ap.error('geometry and dynamics parameters must be positive')
    p = np.loadtxt(args.centreline, delimiter=',', comments='#')[:,:2]
    if np.linalg.norm(p[0]-p[-1]) < 1e-5: p=p[:-1]
    if not np.isfinite(p).all(): raise ValueError('Nonfinite centreline')
    m = TrackMap(args.map, args.seal_gap)
    initial_clear = float(m.clearance(dense_segments(p)).min())
    if initial_clear < args.clearance:
        raise ValueError(f'Centreline clearance {initial_clear:.3f} < requested {args.clearance}; inspect sealed map. Do not silently lower margin.')
    tangent = np.roll(p,-1,axis=0)-np.roll(p,1,axis=0)
    tangent /= np.linalg.norm(tangent,axis=1)[:,None]
    normals = np.column_stack((-tangent[:,1],tangent[:,0]))
    # Reserve more tracking margin in wide corridors, without making narrow
    # sections infeasible. This is map-derived, not a location-specific patch.
    required = np.maximum(args.clearance, np.minimum(.65, m.clearance(p)-.02))
    bounds=[]
    offsets = np.arange(0,args.max_offset+.001,.01)
    for sign in [-1,1]:
        safe = m.clearance(p[:,None,:]+sign*offsets[None,:,None]*normals[:,None,:]) >= required[:,None]+.01
        safe[:,0]=True
        lengths=[]
        for row in safe:
            bad=np.flatnonzero(~row)
            lengths.append(offsets[max(0,int(bad[0])-1)] if len(bad) else offsets[-1])
        bounds.append(np.array(lengths))
    lo,hi=-bounds[0],bounds[1]
    # Freeze zero-width corridors with tiny numerical bounds.
    lo=np.minimum(lo,-1e-8);hi=np.maximum(hi,1e-8)
    n=len(p)
    pattern=lil_matrix((3*n,n))
    for i in range(n):
        pattern[i,[(i-1)%n,i,(i+1)%n]]=1
        pattern[n+i,i]=1
        pattern[2*n+i,[i,(i+1)%n]]=1
    def residual(alpha):
        q=p+alpha[:,None]*normals
        ds,k=geometry(q)
        return np.r_[k*np.sqrt(ds), .06*alpha, .12*(np.roll(alpha,-1)-alpha)]
    fit=least_squares(residual,np.zeros(n),bounds=(lo,hi),jac_sparsity=pattern.tocsr(),
                      max_nfev=400,ftol=1e-8,xtol=1e-8,gtol=1e-7)
    if fit.status < 0 or not np.isfinite(fit.x).all(): raise RuntimeError(f'Optimiser failed: {fit.message}')
    def metrics(q):
        ds,k=geometry(q)
        v=velocity(q,args.max_speed,args.lateral_accel,args.accel,args.brake)
        return dict(length_m=float(ds.sum()), peak_curvature=float(np.abs(k).max()),
                    curvature_energy=float(np.sum(k*k*ds)),
                    ideal_profile_seconds=float(np.sum(2*ds/(v+np.roll(v,-1)))),
                    min_reference_clearance_m=float(m.clearance(dense_segments(q)).min()))
    # Enforce dense polygon clearance and safe homotopy back to official route.
    candidates=[]
    for fraction in np.linspace(1.,0.,21):
        q=p+fraction*fit.x[:,None]*normals
        if m.clearance(dense_segments(q)).min() < args.clearance: continue
        if np.any(m.clearance(q)<required-1e-6): continue
        valid=True
        for t in np.linspace(0,1,31):
            if m.clearance(dense_segments(p+t*(q-p),.04)).min()<args.clearance:
                valid=False; break
        if valid: candidates.append((metrics(q)['ideal_profile_seconds'],q,fraction))
    if not candidates: raise RuntimeError('No geometrically feasible candidate')
    _,q,fraction=min(candidates,key=lambda v:v[0])
    v=velocity(q,args.max_speed,args.lateral_accel,args.accel,args.brake)
    np.savetxt(args.output,np.column_stack((q,v)),delimiter=',',fmt='%.8f',
               header='x_m,y_m,v_mps; original normal-offset curvature optimiser; NOT simulator validated')
    report=dict(arguments=vars(args),solver_success=bool(fit.success),solver_message=fit.message,selected_fraction=float(fraction),
                sealed_component_pairs=len(m.links),baseline=metrics(p),candidate=metrics(q),
                warning='Ideal profiles exclude tracking, tyre dynamics, LiDAR guards and actuator delays. NOT race benchmarks.')
    Path(args.output+'.json').write_text(json.dumps(report,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,10))
        h,w=m.image.shape
        ax.imshow(m.occupied,origin='upper',cmap='gray_r',extent=[m.origin[0],m.origin[0]+w*m.res,m.origin[1],m.origin[1]+h*m.res])
        ax.plot(*p.T,color='#808080',lw=1,label='Official centreline')
        points=ax.scatter(*q.T,c=v,s=9,cmap='plasma',label='Candidate')
        fig.colorbar(points,ax=ax,label='Planned speed (m/s)')
        ax.set_aspect('equal');ax.legend();ax.set_title('Offline candidate — sealed cone barriers')
        fig.savefig(args.output+'.png',dpi=150,bbox_inches='tight');plt.close(fig)
    except ImportError: pass
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
