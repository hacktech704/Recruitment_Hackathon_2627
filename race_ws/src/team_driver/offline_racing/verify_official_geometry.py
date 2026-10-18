#!/usr/bin/env python3
"""Read the installed official cone-linking tool without modifying judging files."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from generate_raceline import TrackMap,dense_segments

def verify(repo, paths, centreline):
    tool=Path(repo)/'scripts/track_tool.py'
    spec=importlib.util.spec_from_file_location('official_track_geometry',tool)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    grid=mod.OccupancyMap(str(Path(repo)/'maps/icra26.yaml'));links=grid.link_cones(1.1)
    m=TrackMap(Path(repo)/'maps/icra26.yaml')
    official=np.array([grid._is_obstacle(v) for v in grid.pixels]).reshape(grid.height,grid.width)
    m.occupied|=official
    m.distance=distance_transform_edt(np.pad(~m.occupied,1,constant_values=False))[1:-1,1:-1]*m.res-m.res*np.sqrt(2)
    p=np.loadtxt(centreline,delimiter=',')[:,:2]
    output={'official_cone_links':links,'files':{}}
    for path in paths:
        q=np.loadtxt(path,delimiter=',')[:,:2]
        if q.shape!=p.shape:raise ValueError('Path correspondence changed')
        clearance=float(m.clearance(dense_segments(q)).min())
        corridor=min(float(m.clearance(dense_segments(p+t*(q-p))).min()) for t in np.linspace(0,1,31))
        if min(clearance,corridor)<.40:raise ValueError(f'{path}: official barrier clearance failed')
        output['files'][str(path)]={'minimum_clearance_m':clearance,'interpolated_corridor_minimum_m':corridor}
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,default=Path('/hackathon'))
    ap.add_argument('--centreline',required=True);ap.add_argument('paths',nargs='+');a=ap.parse_args()
    print(json.dumps(verify(a.repo,a.paths,a.centreline),indent=2))
