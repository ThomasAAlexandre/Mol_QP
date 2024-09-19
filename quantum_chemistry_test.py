from quantum_chemistry import get_evs
from quantum_chemistry_plot import real_vs_estimated, create_3d_rotating


distances, tot_thetas, tot_energy, angles, dists, evs, real_evs = get_evs()
create_3d_rotating(distances, tot_thetas, tot_energy, angles, dists, evs)
real_vs_estimated(distances, evs, real_evs)
