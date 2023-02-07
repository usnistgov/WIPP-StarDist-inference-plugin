
# WIPP StarDist inference plugin  

##  Statements of purpose and maturity
The purpose of this work is to create a [WIPP](https://github.com/usnistgov/WIPP) plugin based on the [StarDist 2D Object detection package](https://github.com/stardist/stardist). 

  
##  Description of the repository contents

- `src`: contains the source Python code
- `Dockerfile`
- `plugin.json` WIPP plugin manifest

###   Technical installation instructions, including operating system or software dependencies

The code is written in Python 3 (tested on version 3.10) and leverages the tensorflow and stardist python packages.

## Installation (optional if using the pre-built Docker image)

### Build Python Virtual Environment 
```
conda create --name stardist python=3.10
conda install grpcio
pip install tensorflow stardist imagecodecs
conda activate stardist
```
	
### Build the Docker image
```
docker build . -t wipp/wipp-stardist-inference-plugin:0.0.1
```
	
## Execution

Pre-trained model [choices from the StarDist package](https://github.com/stardist/stardist#pretrained-models-for-2d) are: `2D_versatile_fluo`, `2D_paper_dsb2018` and `2D_versatile_he`

### Run the Python code

From this directory:
```
python ./src/stardist-inference.py \
--inputImages ./sample-data/images \
--output ./sample-data/outputs
--pretrainedModel 2D_versatile_fluo
```

### Run the Docker image
From this directory, assuming the images to process are in a folder "sample-data/images":
```
docker run -v "$PWD"/sample-data:/data \
wipp/wipp-stardist-inference-plugin:0.0.1 \
--inputImages /data/images \
--output /data/outputs \
--pretrainedModel 2D_versatile_fluo
```
`-v`: mounts a volume/folder from your machine inside of the Docker container

### Run the WIPP plugin
	- register the plugin.json in a deployed WIPP instance - see https://github.com/usnistgov/WIPP
	- upload input images as WIPP image collection
	- create a workflow by adding one step called stardist-inference
	- run and monitor the workflow execution
	- download resulting WIPP image colection

## Additional Information

###    Contact information
-   WIPP team, ITL NIST, Software and System Division, Information Systems Group
-   Contact email address at NIST: wipp-team@nist.gov
 
###    Related Material
-    StarDist Github repository: https://github.com/stardist/stardist

###    Citation: 
Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell Detection with Star-convex Polygons. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.




