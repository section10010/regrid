#!/usr/bin/env python3

import tempfile
import shutil
import os

import numpy as np
import h5py
from matplotlib import pyplot as plt

import damask

load = 'tensionX'
grid = '20grains16x16x16'
mat = 'material'
grid2 = grid+'-2'
grid3 = grid+'-3'

F = []
P = []

cwd = os.getcwd()
print(wd := tempfile.mkdtemp())


def new_cells(F_avg,cells):
    return (F_avg@cells * np.max(cells/(F_avg@cells))).astype(int)


def regrid_restart(fname_in,fname_out,mapping_flat,mapping_phase):
    with (h5py.File(fname_in) as f_in, h5py.File(fname_out,'w') as f_out):
        f_in.copy('homogenization',f_out)

        f_out.create_group('phase')
        for label in f_in['phase']:
            m = mapping_flat[np.isin(mapping_flat,mapping_phase[label])]
            for i,j in enumerate(mapping_phase[label]):
                m[m==j] = i
            f_out['phase'].create_group(label)
            F_e0 = np.matmul(f_in['phase'][label]['F'][()],np.linalg.inv(f_in['phase'][label]['F_p'][()]))
            R_e0, V_e0 = damask.mechanics._polar_decomposition(F_e0, ['R','V'])
            f_out['phase'][label].create_dataset('F',data=np.broadcast_to(np.eye(3),(len(m),3,3,)))
            f_out['phase'][label].create_dataset('F_e',data=R_e0[m])
            f_out['phase'][label].create_dataset('F_p',data=damask.tensor.transpose(R_e0)[m])
            f_out['phase'][label].create_dataset('S',data=np.zeros((len(m),3,3)))
            for d in f_in['phase'][label]:
                if d in f_out[f'phase/{label}']: continue
                f_out['phase'][label].create_dataset(d,data=f_in['phase'][label][d][()][m])

        f_out.create_group('solver')
        for d in ['F','F_lastInc']:
            f_out['solver'].create_dataset(d,data=np.broadcast_to(np.eye(3),np.append(cells_new.prod(),(3,3))))
        for d in ['F_aim', 'F_aim_lastInc']:
            f_out['solver'].create_dataset(d,data=np.eye(3))
        f_out['solver'].create_dataset('F_aimDot',data=np.zeros((3,3)))
        for d in f_in['solver']:
            if d not in f_out['solver']: f_in['solver'].copy(d,f_out['solver'])

# normal run
damask.util.run(f'DAMASK_grid -g {cwd}/{grid}.vti -l {cwd}/{load}.yaml -m {cwd}/{mat}.yaml -w {wd}')
r = damask.Result(f'{wd}/{grid}_{load}_{mat}.hdf5')
F.append([np.average(_,axis=0) for _ in r.place('F').values()])
P.append([np.average(_,axis=0) for _ in r.place('P').values()])
r.add_IPF_color([0,0,1])
r.export_VTK(target_dir=cwd)
F_avg = np.average(r.view(increments=-1).place('F'),axis=0)

# regrid 1
cells_new = new_cells(F_avg,r.cells)
mapping_phase = r._mappings()[0][0]
mapping = damask.grid_filters.regrid(r.size,r.view(increments=-1).place('F').reshape(tuple(r.cells)+(3,3)),cells_new)
mapping_flat = mapping.reshape(-1,order='F')

g = damask.GeomGrid.load(f'{grid}.vti')
g.size = F_avg@g.size
g2 = g.assemble(mapping)
g2.save(f'{wd}/{grid2}.vti')

regrid_restart(f'{wd}/{grid}_{load}_{mat}_restart.hdf5',f'{wd}/{grid2}_{load}_{mat}_restart.hdf5',mapping_flat,mapping_phase)

r.view(increments=0).export_DADF5(f'{wd}/{grid2}_{load}_{mat}.hdf5',mapping=mapping)

with h5py.File(f'{wd}/{grid2}_{load}_{mat}.hdf5','a') as f:
    f['geometry'].attrs['size'] = g.size

shutil.copyfile(f'{cwd}/{load}-2.yaml',f'{wd}/{load}.yaml')
shutil.copyfile(f'{wd}/{grid}_{load}_{mat}.sta',f'{wd}/{grid2}_{load}_{mat}.sta')
damask.util.run(f'DAMASK_grid -g {grid2}.vti -l {load}.yaml -m {cwd}/{mat}.yaml -r 190 -w {wd}')
r = damask.Result(f'{wd}/{grid2}_{load}_{mat}.hdf5').view_less(increments=0)
F.append([np.average(_,axis=0) for _ in r.place('F').values()])
P.append([np.average(_,axis=0) for _ in r.place('P').values()])
r.add_IPF_color([0,0,1])
r.export_VTK(target_dir=cwd)
F_avg = np.average(r.view(increments=-1).place('F'),axis=0)

# regrid 2
cells_new = new_cells(F_avg,r.cells)
mapping_phase = r._mappings()[0][0]
mapping = damask.grid_filters.regrid(r.size,r.view(increments=-1).place('F').reshape(tuple(r.cells)+(3,3)),cells_new)
mapping_flat = mapping.reshape(-1,order='F')

g2.size = F_avg@g2.size
g2.assemble(mapping).save(f'{wd}/{grid3}.vti')

regrid_restart(f'{wd}/{grid2}_{load}_{mat}_restart.hdf5',f'{wd}/{grid3}_{load}_{mat}_restart.hdf5',mapping_flat,mapping_phase)

r.view(increments=0).export_DADF5(f'{wd}/{grid3}_{load}_{mat}.hdf5',mapping=mapping)

with h5py.File(f'{wd}/{grid3}_{load}_{mat}.hdf5','a') as f:
    f['geometry'].attrs['size'] = g2.size

shutil.copyfile(f'{cwd}/{load}-3.yaml',f'{wd}/{load}.yaml')
shutil.copyfile(f'{wd}/{grid2}_{load}_{mat}.sta',f'{wd}/{grid3}_{load}_{mat}.sta')
damask.util.run(f'DAMASK_grid -g {grid3}.vti -l {load}.yaml -m {cwd}/{mat}.yaml -r 440 -w {wd}')
r = damask.Result(f'{wd}/{grid3}_{load}_{mat}.hdf5').view_less(increments=0)
F.append([np.average(_,axis=0) for _ in r.place('F').values()])
P.append([np.average(_,axis=0) for _ in r.place('P').values()])
r.add_IPF_color([0,0,1])
r.export_VTK(target_dir=cwd)

# plot average stress-strain curve
F_ = np.concatenate([F[0],[F[0][-1]@F1 for F1 in F[1]], [F[0][-1]@F[1][-1]@F2 for F2 in F[2]]])

epsilon = damask.mechanics.strain(F_,m=0.0,t='V')
sigma = damask.mechanics.stress_Cauchy(np.concatenate([P[0],P[1],P[2]]),
                                       np.concatenate([F[0],F[1],F[2]]))

fig, ax1 = plt.subplots()

ax1.set_xlabel('strain')
ax1.set_ylabel('Cauchy stress / Pa')
ax1.plot(epsilon[:,0,0],sigma[:,0,0])

plt.show()
