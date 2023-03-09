<p align="left">
	<img src="img/DG-logo.png" alt="Digital-Gaia" width="200">
</p>

# Welcome to GaiaHub
#### *Digital Gaia's Global Sustainability Information and Collaboration Hub*

Designed as an experimental environment for **GaiaHub** contributors who are working primarily on 
model and engine improvements.

## Table of contents

1. [Getting started](#1-getting-started)
2. [Recommended installation process](#2-recommended-installation-process)
3. [Roots & Culture Indoor Agriculture Modeler's Lab](#3-roots--culture-indoor-agriculture-modelers-lab)
4. [Outdoor Agriculture Modeler's Lab](#4-outdoor-agriculture-modelers-lab)
5. [Agroforestry Modeler's Lab](#5-agroforestry-modelers-lab)
6. [How to contribute to GaiaHub](#6-how-to-contribute-to-gaiahub)
7. [Provenance policy](#7-provenance-policy)
8. [Known errors](#8-known-errors)
9. [Citing GaiaHub](#9-citing-gaiahub)

### 1. Getting started
**GaiaHub** is a place where scientists, coders, 
citizens of earth and other forward-thinking creatures come together to co-develop
an open model library describing all the world’s agriculture, to help 1 billion 
farmers become more financially and environmentally sustainable.

If you are new or if you haven't yet, check out the [Get Started notebook](https://github.com/gaia-os/gaia-hub/blob/main/notebooks/get_started.ipynb) 
for a detailed breakdown of the engine and how to harness it's power!

Once you are oriented, check out the [Roots & Culture Indoor Agriculture Modeler's Lab](#3-roots--culture-indoor-agriculture-modelers-lab) 
section below for more instructions on how to use notebooks for conducting your own experiments. If you are a contributor looking to complete a project this will 
be a useful tool for demonstrating how your contributions improve or expand the engine.

When you are ready to propose merging your changes into the core engine, check out [these tips](#6-how-to-contribute-to-gaiahub).

###  2. Recommended installation process
1. Clone this repository and the fangorn submodule into a folder on your local system using `terminal` (macOS and Linux) or `cmd` (Windows):
    ```
    git clone --recurse-submodules -b main https://github.com/gaia-os/gaia-hub.git
    ```
    Replace `main` with the name of any branch you'd like to clone.  
    Fangorn is the name of the Digital Gaia core engine, and contains many important files including the ontology, agents and more.
2. Move back to project root  
3. Use the following commands to create a new virtual environment and then activate it:  
**Note**: first [Install virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) if not already installed
   ```
    $ python3 -m venv myvenv
    ```
    This will create the virtual Python environment called `myenv`. Replace `myenv` with a different name if you prefer.
    ```
    $ source myenv/bin/activate
    ```
    This will activate the virtual environment called `myenv`.  
4. Install all dependencies for the Digital Gaia engine  
    ```
    (myenv) $ python -m pip install -r requirements.txt
    ```
   Note the `(myenv)` which indicates you are indeed working in an activated virtual environment.  
5. Open a jupyter notebook and start exploring the `notebooks/` folder  

### 3. Roots & Culture Indoor Agriculture Modeler's Lab
Roots & Culture operates a state-of-the-art indoor hemp farm in Virginia, growing oil rich hemp in soil under grow lights. 

Using a curated, provenanced dataset from their farm we have built a standard environment in which one can innovate and demonstrate how one's proposed model changes 
effect predictions and inference. This modeler's lab of sorts is a Jupyter notebook that presents a simple example of how one can 
demonstrate how changes to the model effect predictions. 

We hope this well-characterized model farm will serve as a jumping off point for modeler's looking to quickly make and test 
their own contributions. [Check it out here](https://github.com/gaia-os/gaia-hub/blob/main/notebooks/LAB-roots-and-culture.ipynb)!

### 4.Outdoor Agriculture Modeler's Lab
Do you have expert knowledge about a crop category and access to detailed field data for that crop? If so, help us build the 
Outdoor Agriculture Modeler's Lab!

Stay tuned for updates on when this lab will become available. When it is, this lab will be used for developing aspects of the model that 
are not possible to meaningfully model in an indoor facility (e.g. sunlight, weather, etc.).

### 5. Agroforestry Modeler's Lab
Do you have expert knowledge about a forest product category and access to detailed agroforestry data? If so, help us build the 
Agroecology Modeler's Lab!

Stay tuned for updates on when this lab will become available. Like the Outdoor Lab, this lab will make it possible to meaningfully test 
model developments that deal with concepts which apply only to agroforestry projects (e.g. specialty crops like cacao, impact of koalas on vegetation, etc.).

### 6. How to contribute to GaiaHub
First, fill out [this form](https://forms.gle/E1C8QAKJio4ParXm8) to join **GaiaHub** and get access to our Slack channel.

If you're looking to improve the model by contributing your own code please use a basic 
[fork and pull request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) protocol for 
fastest and best integration with our current internal workflow:
1. Fork the `gaia-hub` repository into your personal GitHub account 
2. Clone the branch you want to work from into a folder on your local system
3. Work on a new branch you create
4. When you're ready to merge your changes into the `gaia-hub` create a pull request

### 7. Provenance policy
All structural assumptions and parameter value priors (growth rates, effect sizes, etc) in the library should be given provenance. 
Order of preference for provenance:
1. Estimated from reference datasets that have their own provenance, with snapshots and URIs in the model library repository
2. Adapted from reference papers or articles available on the Web, with snapshots and URIs referenced in the repository
3. Reported by identifiable experts through written communication, with stable URI (ex: Google Docs copy of email exchange) referenced in the repository

### 8. Known errors
1. JAX on Apple Silicon (M1 and M2 series chips)
While [some people have reported success](https://stackoverflow.com/questions/70815864/how-to-install-trax-jax-jaxlib-on-m1-mac-on-macos-12) 
installing the [CPU-only version of JAX](https://github.com/google/jax#installation) or [building JAX from source](https://jax.readthedocs.io/en/latest/developer.html), 
we have yet to indentify a reliable workaround. 
Please [contact us](https://www.digitalgaia.earth/#Contact) if this presents an impasse.

### 9. Citing GaiaHub
To cite this repository:
```
@software{GaiaHub2023github,
  author = {Digital Gaia GaiaHub Network},
  title = {{GaiaHub}: Digital Gaia's Global Sustainability Information and Collaboration Hub},
  url = {http://github.com/gaia-os/gaia-hub},
  version = {0.1.0},
  year = {2023},
}
```

