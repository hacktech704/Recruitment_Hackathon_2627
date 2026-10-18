#!/usr/bin/env python3
"""Reproduce both trajectories from official map/track metadata, outside protected paths."""
import argparse,subprocess,sys,json
from pathlib import Path
from verify_official_geometry import verify

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--repo',type=Path,default=Path('/hackathon'))
ap.add_argument('--output-dir',type=Path,default=None)
a=ap.parse_args();repo=a.repo.resolve();here=Path(__file__).resolve().parent
out=(a.output_dir or repo/'race_ws/src/team_driver/config').resolve()
for protected in ['maps','scripts','install','docker','external','.github','race_ws/src/roboracer_referee']:
    if out.is_relative_to(repo/protected):raise SystemExit('Refusing output inside protected judging directory')
out.mkdir(parents=True,exist_ok=True)
centre=out/'generated_centreline.csv';curve=out/'icra26_raceline.csv';timed=out/'icra26_time.csv'
commands=[
 [sys.executable,str(repo/'scripts/track_tool.py'),'centerline','--output',str(centre)],
 [sys.executable,str(here/'generate_raceline.py'),'--map',str(repo/'maps/icra26.yaml'),'--centreline',str(centre),'--output',str(curve)],
 [sys.executable,str(here/'refine_time.py'),'--map',str(repo/'maps/icra26.yaml'),'--centreline',str(centre),'--seed',str(curve),'--output',str(timed)]]
for cmd in commands:subprocess.run(cmd,check=True,cwd=repo)
report=verify(repo,[curve,timed],centre)
(out/'official_geometry_check.json').write_text(json.dumps(report,indent=2))
print('Generated and checked both trajectories:',out)
