<p align="left">
	<img src="img/DG-logo.png" alt="Digital-Gaia" width="200">
</p>

# The Open Science Network Modeler's Labs
Designed as an experimental environment for **Open Science Network (OSN)** contributors who are working primarily on 
model and engine improvements.

### Table of contents

1. [Getting started](#getting-started-and-welcome-to-the-osn)
2. [Recommended installation process](#recommended-installation-process)
3. [How to contribute to the Open Science Network](#how-to-contribute-to-the-open-science-network)
4. [Roots & Culture Indoor Agriculture Modeler's Lab](#roots--culture-indoor-agriculture-modelers-lab)
5. [Outdoor Agriculture Modeler's Lab](#outdoor-agriculture-modelers-lab)
6. [Agroecology Modeler's Lab](#agroecology-modelers-lab)
5. [Provenance policy](#provenance-policy)

## Getting started and Welcome to the OSN
The **Open Science Network (OSN)** is a place where scientists, coders, 
citizens of earth and other forward-thinking creatures come together to co-develop
an open model library describing all the world’s agriculture, to help 1 billion 
farmers become more financially and environmentally sustainable.

If you haven't yet, check out the `get-started` notebook for a detailed breakdown of the engine.

Once you are oriented, check out the [Roots & Culture Indoor Agriculture Modeler's Lab](#roots--culture-indoor-agriculture-modelers-lab) section for more instructions on how to use notebooks 
for conducting your own experiments.

When you are ready to request merging your proposed changes into the core engine, check out these tips: [How to contribute to the **Open Science Network**](#recommended-installation-process).

## Recommended installation process
1. Clone into a folder on your local system:
    ```
    git clone -b main https://github.com/gaia-os/open-science-network.git
    ```
    Replace `main` with the name of any branch you'd like to clone.  
2. [Install virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)  
3. Move to project root  
4. Use the following commands to create a new virtual environment and then activate it:  
    ```
    $ python3 -m venv myvenv
    ```
    This will create the virtual Python environment called `myenv`. Replace `myenv` with a different name if you prefer.
    ```
    $ source myenv/bin/activate
    ```
    This will activate the virtual environment called `myenv`.  
5. Install all dependencies for the Digital Gaia engine
    ```
    (myenv) $ python -m pip install -r requirements.txt
    ```
   Note the `(myenv)` which indicates you are indeed working in an activated virtual environment.  
6. Open a jupyter notebook and start exploring the `notebooks/` folder

## Roots & Culture Indoor Agriculture Modeler's Lab
Roots & Culture operates a state-of-the-art indoor hemp farm in Virginia, growing high cannabinoid (CBD, CBG, etc.) hemp in soil under grow lights. 

Using data from their farm we have built a standard environment in which you can innovate and demonstrate how your proposed model changes 
effect predictions and inference. This modeler's lab of sorts is a Jupyter notebook that presents a simple example of how one might 
demonstrate how changes to the model effect predictions. We hope it will serve as a jumping off point for modeler's looking to quickly make 
their own contributions.

## Outdoor Agriculture Modeler's Lab
Do you have expert knowledge about a crop category and access to detailed field data for that crop? If so, help us build the 
Outdoor Agriculture Modeler's Lab!

Stay tuned for updates on when this lab will become available.

## Agroecology Modeler's Lab
Do you have expert knowledge about a forest product category and access to detailed agroforestry data? If so, help us build the 
Agroecology Modeler's Lab!

Stay tuned for updates on when this lab will become available.

## How to contribute to the Open Science Network
First, fill out [this form](https://forms.gle/E1C8QAKJio4ParXm8) to join the **OSN** and get access to our Slack channel.

If you're looking to improve the model by contributing your own code please use a basic 
[fork and pull request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) protocol for 
fastest and best integration with our current internal workflow:
1. Fork the `open-science-network` repository into your personal GitHub account 
2. Clone the branch you want to work from into a folder on your local system
3. Work on a new branch you create
4. When you're ready to merge your changes into the `open-science-network` create a pull request

## Provenance policy
All structural assumptions and parameter value priors (growth rates, effect sizes, etc) in the library should be provenanced. 
Order of preference for provenance:
1. Estimated from reference datasets that are themselves provenanced, with snapshots and URIs in the model library repository
2. Adapted from reference papers or articles available on the Web, with snapshots and URIs referenced in the repository
3. Reported by identifiable experts through written communication, with stable URI (ex: Google Docs copy of email exchange) referenced in the repository

