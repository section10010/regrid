#! /usr/bin/env python3

import damask
import numpy as np

damask.util.run(f'DAMASK_grid -g simple.vti -l tensionX_short.yaml -m material.yaml')

r = damask.Result('simple_tensionX_short_material.hdf5').view(increments=0)
cells_new = (4,6,1)

print('global indices:\n',idx := np.arange(np.prod(r.cells)).reshape(r.cells,order='F')[:,:,0])

mapping_phase = r._mappings()[0][0]
for phase in r.phases:
    print(f'mask {phase}\n',np.isin(idx,mapping_phase[phase]))

F_avg = np.average(r.place('F'),axis=0)
mapping = damask.grid_filters.regrid(r.size,r.view(increments=-1).place('F').reshape(tuple(r.cells)+(3,3)),cells_new)
mapping_flat = mapping.reshape(-1,order='F')

print(f'\nregridding from {r.cells} to {cells_new}')
print('global mapping:\n',mapping[:,:,0],'\n')

for phase in r.phases:
    print(f'mask {phase}\n',np.isin(mapping[:,:,0],mapping_phase[phase]))

for phase in r.phases:
    m = mapping_flat[np.isin(mapping_flat,mapping_phase[phase])]
    print(f'global incides for {phase}\n',m)
    for i,j in enumerate(mapping_phase[phase]):
        m[m==j] = i
    print(f'local incides for {phase}\n',m,'\n')
