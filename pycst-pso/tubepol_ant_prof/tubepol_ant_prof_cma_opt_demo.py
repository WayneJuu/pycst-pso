import sys
import numpy as np
import cma

from tubepol_ant_prof_pycst_cfg import *
from pycst_proj_simulator import PyCstProjSimulator

# Start CMA-ES optimization
horn_sim = PyCstProjSimulator(proj_cfg_dic, data_analyser_cfg_dic, optimizer_cfg_dic)
# horn_sim.func_cst_proj_sim(cma_init_para_list)
# horn_sim.optimizer_ins.optimize(horn_sim.func_cst_proj_sim)
# best_params, best_cost = horn_sim.optimizer_ins(horn_sim.func_cst_proj_sim)
best_params, best_cost = horn_sim.run_pso_optimization(100)
