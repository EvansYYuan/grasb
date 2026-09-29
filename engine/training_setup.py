# code adapted from https://github.com/matteopariset/unbalanced_sb/tree/main/udsb_f

import jax
import haiku as hk
from functools import partial

import pickle
import datetime

from utils import *
from datasets import *
from models import *
from sde import *


class Training_Setup:
    def __init__(self, dataset: Input_Dataset, dataset_name="Data_1", steps_num=100, epochs = 5, vae_epochs=100, key=None, params=None, objective="divergence",
                 batch_size=512, hidden_dim=[64], dec_hidden_size=[64], ferryman_hidden_dim=[64], ferryman_activate_final=True,
                 ipf_mask_dead=False, reality_coefficient=0.1, paths_reuse=5, num_sde=10, resnet=False,
                 feature_spatial_loss=False, t_dim=16, vae_input_dim=1000, vae_enc_hidden_dim=[512,512],
                 vae_dec_hidden_dim =[512,512], vae_t_dim=8, calc_latent_loss=True, calc_recon_loss=True,
                 vae_latent_dim=64, vae_batch_size=64, killer_func=Input_Dataset.killing_function,
                 no_vae=False, grn=None, lambda_grn=0.0, grn_time_scale=None,
                 grn_latent=False, grn_dec_fn=None, grn_encjvp_fn=None,
                 grn_penalty="l2"):

        # GRN-USB pull-back (plan §2): when `no_vae=True` the SB + Ferryman run directly in
        # HVG gene space (the original udsb_f data-space wiring), with NO VAE encode/decode.
        # The SB state dimension is then the gene count (`dataset.input_dim`) instead of the
        # VAE latent dim, and the VAE latent/recon auxiliary losses are switched off. The
        # ARTEMIS scalar-potential score (Z = g·∇φ) is kept unchanged — only its input dim
        # changes. λ→0 of the GRN penalty (Step 3) recovers this engine exactly.
        self.no_vae = no_vae
        if no_vae:
            calc_latent_loss = False
            calc_recon_loss = False

        self.dataset_name=dataset_name
        self.dataset=dataset
        self.steps_num = dataset.steps_num
        self.epochs = epochs
        self.vae_epochs = vae_epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.dec_hidden_size = dec_hidden_size
        self.ipf_mask_dead = ipf_mask_dead
        self.objective=objective
        self.reality_coefficient = reality_coefficient # Birth by splitting are not accounted for when reality_coefficient is big (i.e. alive_to_alive should contribute to births)
        self.paths_reuse = paths_reuse
        self.num_sde = num_sde
        self.resnet = resnet
        self.t_dim=t_dim
        self.vae_input_dim=vae_input_dim
        self.vae_enc_hidden_dim=vae_enc_hidden_dim
        self.vae_dec_hidden_dim=vae_dec_hidden_dim
        self.vae_t_dim=vae_t_dim
        self.vae_latent_dim=vae_latent_dim
        self.vae_batch_size=vae_batch_size
        self.ferryman_hidden_dim=ferryman_hidden_dim
        self.calc_latent_loss=calc_latent_loss
        self.calc_recon_loss=calc_recon_loss
        #self.killer_func = killer_func

        # SB state dimension: gene space when the VAE is removed, else the VAE latent dim.
        # Used by Trainer to init the forward/backward drift nets at the right width.
        self.state_dim = dataset.input_dim if no_vae else vae_latent_dim

        # --- GRN drift (λ-leg, plan §8). `grn` is a grn_drift.GRNDrift; lambda_grn=0 ⇒ off.
        # Precompute the JAX lookups the loss closes over: per-particle bin = (cell type,
        # interval(step)), and the interval→[0,1]-integration time-scale conversion. ---
        import jax.numpy as jnp
        self.grn = grn
        self.lambda_grn = float(lambda_grn or 0.0)
        # --- penalty FORM (2026-07-15). "l2" = ||b_θ − b_GRN||²: one term pinning BOTH the
        # direction and the magnitude of the drift. The gamma scan showed that is the problem:
        # biology reads only b_GRN's DIRECTION (recall flat 0.40 across a 29× gamma range),
        # while the W2 cost comes from the same term forcing b_θ onto b_GRN's MAGNITUDE, which
        # is ~5× smaller than the observed weekly displacement => the bridge undershoots.
        # No gamma fixes it (damping is inert, amplifying costs +0.5). "cosine" = 1 − cos(b_θ,
        # b_GRN): steers direction and leaves magnitude to the transport objective.
        self.grn_penalty = str(grn_penalty)
        # --- latent-GRN mode (latent_grn/ experiment). When grn_latent=True the SB runs
        # in a FROZEN VAE latent, so the gene-space b_GRN must be pushed into latent before
        # the consistency penalty: decode the latent trajectory pos -> gene x_hat (grn_dec_fn),
        # evaluate gene-space b_GRN(x_hat), then pushforward via the encoder Jacobian
        # (grn_encjvp_fn: (x,v)->J_enc(x).v, already scaled to standardized-latent units).
        # Both closures close over the frozen VAE params; None => gene-space penalty (default).
        self.grn_latent = bool(grn_latent)
        self.grn_dec_fn = grn_dec_fn
        self.grn_encjvp_fn = grn_encjvp_fn
        if grn is not None and self.lambda_grn > 0.0:
            grn.to_jax()
            self.grn_bin_lookup = jnp.asarray(grn.bin_lookup)                 # (n_types, n_intervals)
            self.grn_step_to_interval = jnp.asarray(grn.step_to_interval(self.steps_num))
            # dev span maps onto [0,1]; one unit-interval = 1/n_intervals of integration time,
            # so a per-interval GRN drift is n_intervals× a per-[0,1] drift (matches b_θ units).
            self.grn_time_scale = float(grn.n_intervals if grn_time_scale is None else grn_time_scale)
        else:
            self.grn_bin_lookup = None
            self.grn_step_to_interval = None
            self.grn_time_scale = 1.0

        #assert self.hidden_dim > dataset.input_dim, "The network hidden size should be bigger than the dimension of the state space"        

        mass_max = max(dataset.mass)

        
        # 2. define SDE class
        self.sde = SDE(dataset=dataset, steps_num=self.steps_num, batch_size=self.batch_size)

        # 3. sample marginals (training.py, viewer.py)
        self.start_marginals_sampler = { FORWARD: partial(dataset.pi_0_sample, n_samples = self.batch_size),
                                   BACKWARD: partial(dataset.pi_1_sample, n_samples = self.batch_size)}
        
        # 4a. define models: Forward, Backward
        self.model = { FORWARD: hk.transform(init_base_model( self.hidden_dim, dec_hidden_size=self.dec_hidden_size, 
                                                             resnet=self.resnet, t_dim=self.t_dim)), 
                 BACKWARD: hk.transform(init_base_model(self.hidden_dim,  dec_hidden_size=self.dec_hidden_size, t_dim=self.t_dim))}

        # 4b. define ferryman model
        self.ferryman = hk.transform(init_ferryman_model(ferryman_hidden_dim, ferryman_activate_final))


        self.vae_model = hk.transform(lambda t,x: VariationalAutoEncoder(self.vae_input_dim, self.vae_enc_hidden_dim,
                                                                         self.vae_dec_hidden_dim, 
                                                                         self.vae_latent_dim, self.vae_t_dim)(t,x))
        
        self.dec = hk.transform(lambda z: Decoder(output_shape=self.vae_input_dim, hidden_size=self.vae_dec_hidden_dim)(z))
        
        #self.optimizer = optax.adam(self.learning_rate)


        # initialize training state
        self.state = (key,params)

    @staticmethod
    def density(model):
        return model.apply

    def score(self, model):
        #Z = lambda params, key, t, pos: self.sde.g(t, pos) * model.apply(params, key, t, pos)
        # return Z
    
        log_density = Training_Setup.density(model)
        grad_log_density = jax.grad(log_density, argnums=3)
         
        grad_log_density = jax.vmap(grad_log_density, in_axes=(None, None, None, 0), out_axes=(0))

        Z = lambda params, key, t, pos: self.sde.g(t, pos) * grad_log_density(params, key, t, pos)

        return Z

    def save(self):
        tag = self.dataset_name + "_" + datetime.datetime.today().strftime('%Y_%m_%d-%H_%M_%S')
        file = open(tag+".pkl","wb")
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self, dataset_name):
        file = open(dataset_name+".pkl","rb")
        obj = file.read()
        file.close()
        self.__dict__ = pickle.loads(obj)


