# code adapted from https://github.com/matteopariset/unbalanced_sb/tree/main/udsb_f

import jax
import jax.random as random
import jax.numpy as jnp
from jax.tree_util import tree_map

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils import *

class Input_Dataset():

    def __init__(self, x, meta, meta_celltype_column=None, splitting_births_frac=0.2, eps=1e-7,steps_num=100,
                 val_split=False, death_importance_rate=100, f_val=None, std_threshold=2., cutoff=0.2, mb_prior=5.0,
                 g_type="triangular", g_max=1.0, cell_type_ids=None):

        # SDE base-drift / diffusion scale. ARTEMIS's defaults (constant base drift f_val and a
        # triangular diffusion peaking at 1.0) were calibrated for the VAE *latent* space. In raw
        # log1p HVG gene space (no_vae) the data per-coord std is ~0.15-0.28 and the real
        # inter-timepoint drift is ~0.03-0.07/coord, so a constant f and a g~1 swamp the signal and
        # the marginals walk off. `g_type`/`g_max` make the diffusion configurable so it can be
        # matched to gene-space scale; set f_val=0 there and let the learned score carry the drift.
        self.g_type = g_type
        self.g_max = g_max

        self.val_split = val_split
        self.death_importance_rate = death_importance_rate
       
        self.mb_prior = mb_prior

        if val_split:
            x,meta,x_val, meta_val = self.split_train_test(x,meta)
            self.x_val, self.meta_val = x_val, meta_val
        else:
            self.x_val, self.meta_val = None, None

        self.x = x.sort_values(by=["time"])
        self.features = x.columns[:-1]
        self.input_dim = len(self.features)
        self.eps = eps
        
        self.meta = meta
        self.meta_celltype_column = meta_celltype_column
        self.steps_num=steps_num

        self.times = self.x["time"].unique().tolist()
        self.times_orig = self.x["time"].unique().tolist()

        self.mass = self.x["time"].value_counts().sort_index().to_list()
        self.splitting_births_frac = splitting_births_frac
        self.f_val=f_val

        #self.time_x_groundtruth = {int(self.cells_time(k)*self.steps_num): jnp.array(x[x["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times}
        self.time_x = {int(self.cells_time(k)*self.steps_num): jnp.array(x[x["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times}
        self.mean = tree_map(lambda x_temp: jnp.array(x_temp.mean(axis=0, keepdims=True)), self.time_x)
        self.std = tree_map(lambda x_temp: jnp.array(x_temp.std(axis=0, keepdims=True)), self.time_x)

        # Per-timepoint cell-type label ids (λ-leg, plan §7), aligned row-for-row to time_x so
        # the label samplers below can reuse the position samplers' permutation. Built with the
        # SAME `x[x["time"]==k]` mask as time_x (assumes val_split=False, the no_vae/GRN path).
        self.cell_type_ids = cell_type_ids
        if cell_type_ids is not None:
            lab = pd.Series(np.asarray(cell_type_ids), index=x.index)
            self.labels_by_step = {int(self.cells_time(k)*self.steps_num):
                                   jnp.asarray(lab[x["time"]==k].to_numpy()) for k in self.times}
        else:
            self.labels_by_step = None

        if self.val_split:
            self.time_x_val = {int(self.cells_time(k)*self.steps_num): jnp.array(x_val[x_val["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times}
    
        self.times = list(map(lambda t: int(self.cells_time(t)*self.steps_num), self.times))

        self.std_threshold = std_threshold
        self.cutoff = cutoff
    
    def update_data_info(self,x, x_val):

        self.time_x = {int(self.cells_time(k)*self.steps_num): jnp.array(x[x["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times_orig}
        if self.val_split:
            self.time_x_val = {int(self.cells_time(k)*self.steps_num): jnp.array(x_val[x_val["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times_orig}

        self.time_x = {int(self.cells_time(k)*self.steps_num): jnp.array(x[x["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times_orig}
        self.mean = tree_map(lambda x_temp: jnp.array(x_temp.mean(axis=0, keepdims=True)), self.time_x)
        self.std = tree_map(lambda x_temp: jnp.array(x_temp.std(axis=0, keepdims=True)), self.time_x)
        
    def split_train_test(self,x,meta):
        X = np.arange(x.shape[0])
        y = x.values[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=104,test_size=0.1, shuffle=True)

        if meta is not None:
            return x.iloc[X_train],meta.iloc[X_train], x.iloc[X_test], meta.iloc[X_test]
        return x.iloc[X_train],None, x.iloc[X_test], None
 
    def f(self,t,x):
        "Returns base drift"
        if self.f_val is not None:
            return self.f_val
        else:
            return 5
    
    def g(self,t,x, type=None):
        "Returns diffusion coefficient"

        if type is None:
            type = self.g_type
        g_max = self.g_max

        if type=="triangular":
            #return triangular_diffusivity(1.5,1.)
            return triangular_diffusivity(t, g_max)
        elif type=="inverse_triangular":
            return inverse_triangular_diffusivity(t, g_max=g_max)
        elif type=="decreasing":
            return decreasing_diffusivity(t, g_max=g_max)
        elif type=="constant":
            return constant_diffusivity(t, g_max=g_max)
    
    def cells_time(self,t):
        return (t-self.times_orig[0]) / (self.times_orig[-1] - self.times_orig[0])
    
    def real_time(self,t):
        return (t*(self.times_orig[-1]-self.times_orig[0]) + self.times_orig[0])
    
    def sample_labels(self, direction, key, n_samples):
        """Cell-type label ids for the particles `pi_{0,1}_sample` draws with the SAME key.

        Reproduces those samplers' `random.permutation(key, tot)[:n]` (meta_celltype_column
        is None on the no_vae/GRN path) so labels align row-for-row with the sampled
        positions. `direction`=FORWARD samples the t0 marginal, BACKWARD the t-last one.
        """
        step = self.times[0] if is_forward(direction) else self.times[-1]
        src = self.labels_by_step[step]
        tot = src.shape[0]
        sel = random.permutation(key, tot)[:min(n_samples, tot)]
        return src[sel]

    def pi_0_sample(self, key, n_samples=300, get_metadata=False):
        "sample cells from initial distribution"

        pi_0_source = self.time_x[self.times[0]]
        tot_samples = pi_0_source.shape[0]

        if self.meta_celltype_column == None:
            sel_idxs = random.permutation(key, tot_samples)[:min(n_samples, tot_samples)]
            return pi_0_source[sel_idxs]
        else:
            meta_t_0 = self.meta[self.meta["time"] == self.times_orig[0]]
            celltype_proportion = ((meta_t_0[self.meta_celltype_column].value_counts()/tot_samples)*min(n_samples, tot_samples)).astype(int)
            samples_per_celltype = celltype_proportion.to_dict()
            df_measurements  = pd.DataFrame(pi_0_source, index = meta_t_0.index)

            list_sampled_indexes = []

            for name, group in meta_t_0.groupby(self.meta_celltype_column):    
                n_rows_to_sample = samples_per_celltype[name]
                sampled_group = group.sample(n_rows_to_sample)
                list_sampled_indexes.extend(sampled_group.index.to_list())

            p1_0_source_sub = jnp.array(df_measurements.loc[list_sampled_indexes].to_numpy().astype(float))

            if get_metadata:
                return p1_0_source_sub, meta_t_0.loc[list_sampled_indexes]
            else:
                return p1_0_source_sub

    def pi_1_sample(self, key, n_samples=300, get_metadata=False):
        "sample cells from terminal distribution"

        pi_1_source = self.time_x[self.times[-1]]
        tot_samples = pi_1_source.shape[0]

        if self.meta_celltype_column == None:
            sel_idxs = random.permutation(key, tot_samples)[:min(n_samples, tot_samples)]
            return pi_1_source[sel_idxs]
        else:
            meta_t_1 = self.meta[self.meta["time"] == self.times_orig[-1]]
            celltype_proportion = ((meta_t_1[self.meta_celltype_column].value_counts()/tot_samples)*min(n_samples, tot_samples)).astype(int)
            samples_per_celltype = celltype_proportion.to_dict()
            df_measurements  = pd.DataFrame(pi_1_source, index = meta_t_1.index)

            list_sampled_indexes = []

            for name, group in meta_t_1.groupby(self.meta_celltype_column):    
                n_rows_to_sample = samples_per_celltype[name]
                sampled_group = group.sample(n_rows_to_sample)
                list_sampled_indexes.extend(sampled_group.index.to_list())

            p1_1_source_sub = jnp.array(df_measurements.loc[list_sampled_indexes].to_numpy().astype(float))
            if get_metadata:
                return p1_1_source_sub, meta_t_1.loc[list_sampled_indexes]
            else:
                return p1_1_source_sub

    def killing_function(self):
        
        density_kernel = gaussian_kernel()

        def _calc_threshold_violation_score(t, t_min, t_max, x):
            effecttive_t = (t-(t_min/self.steps_num)) / ((t_max/self.steps_num) - (t_min/self.steps_num))
            effective_mean = effecttive_t * self.mean[t_max] + (1-effecttive_t) * self.mean[t_min]
            effective_std = effecttive_t * self.std[t_max] + (1-effecttive_t) * self.std[t_min]
            threshold_violation_score = jnp.mean(jnp.abs(x-effective_mean) > self.std_threshold * effective_std, axis=1)
            return (threshold_violation_score < self.cutoff) * 0 + (threshold_violation_score >= self.cutoff)*threshold_violation_score
     
        # define prior kill-rate factor
        delta_t = jnp.array(self.times[1:]) - jnp.array(self.times[:-1])
        self.death_importance = (self.death_importance_rate)*(-1)*(1/delta_t)*jnp.log(jnp.array(self.mass[1:])/jnp.array(self.mass[:-1]))
        
        def _killer(t, x, db_prior=10):

            mean_based_transitions = 0.0
            density_based_transitions = 0.0

            for i in range(1,len(self.times)):
                t_min, t_max = self.times[i-1], self.times[i]
               
                mean_based_transitions += self.death_importance[i-1]*((t_min/self.steps_num) <= t)*(t< (t_max/self.steps_num)) * _calc_threshold_violation_score(t,t_min,t_max,x)
                density_based_transitions += ((t_min/self.steps_num) <= t)*(t< (t_max/self.steps_num)) * jnp.square(kde(density_kernel,self.mean[t_max],x)) * -1
                
            density_based_transitions = density_based_transitions.reshape(density_based_transitions.shape[0],)
            kill_rate = self.mb_prior*mean_based_transitions +density_based_transitions

            return kill_rate
         
        return _killer
    
    


    
