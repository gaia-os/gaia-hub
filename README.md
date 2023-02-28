<p align="left">
	<img src="img/DG-logo.png" alt="Digital-Gaia" width="200">
</p>

# The Roots & Culture Indoor Agriculture Modeler's Lab
This branch is designed as a experimental environment for **OSN** contributors who are working primarily on 
model and engine improvements.

### Table of contents

1. [Getting started](#getting-started)
2. [Recommended installation process](#recommended-installation-process)
3. [How to contribute to the **Open Science Network**](#recommended-installation-process)
4. [Provenance policy](#provenance-policy)

## Getting started
If you haven't yet, check out the get-started notebook for a detailed breakdown of the engine 
and how to, well, get started.

Once you are oriented, check out the `LAB-roots-and-culture` notebook for more instructions on how to use this branch 
for conducting your own experiments.

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