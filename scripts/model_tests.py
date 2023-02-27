from jax import random
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed
from numpyro.infer import Predictive
from numpyro.infer import SVI, Trace_ELBO
import pandas as pd
from enumerations.ForestLots import ForestLots

# Sanity checks
params = dict(
    lot_name=["Control lot", "Just remove", "Remove then plant once", "Remove then plant twice", "Restore"],
    lot_area=[1.0, 1.0, 1.0, 1.0, 1.0],
    action_0=[0, 1, 1, 1, 3],
    action_1=[0, 0, 2, 2, 3],
    action_2=[0, 0, 0, 2, 3]
)
params = pd.DataFrame(params)
lots = ForestLots(params.lot_area)

# Can sample from model
with seed(rng_seed=42):
    model = ForestLots.make_model(lots, params)
    model()

# Prior predictive distribution makes sense
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
prior_predictive = Predictive(model, num_samples=1000)
prior_predictions = prior_predictive(rng_key_)
numpyro.diagnostics.print_summary(prior_predictions, prob=0.89, group_by_chain=False)

# Generate sample observations as the means of the prior predictions
sr = {}
for t in range(lots.T):
    sr_t = prior_predictions[f"obs_{t}"].mean(axis=0)
    sr[f"obs_{t}"] = sr_t

# Can sample from guide
with seed(rng_seed=42):
    guide = ForestLots.make_guide(lots, params)
    guide(sr)

# Fit guide to observations
optimizer = numpyro.optim.Adam(step_size=5e-3)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(random.PRNGKey(0), 10000, obs=sr)

# Prior predictive distribution makes sense
posterior_params = svi.get_params(svi_result.state)
posterior_predictive = Predictive(guide, params=posterior_params, num_samples=1000)
posterior_samples = posterior_predictive(random.PRNGKey(1), sr)
numpyro.diagnostics.print_summary(posterior_samples, prob=0.89, group_by_chain=False)

# Check fit of variational params
print(posterior_params)

A_true_pinv = jnp.linalg.pinv(lots.A_true)
b_true_pinv = -A_true_pinv @ lots.b_true
print("Fit for A:", jnp.linalg.norm(posterior_params['Ainv'] - A_true_pinv, 2))
print("Fit for b:", jnp.linalg.norm(posterior_params['binv'] - b_true_pinv, 2))
