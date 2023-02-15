# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

from stardist.models import StarDist2D, StarDist3D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
from tifffile import imread, imwrite
from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import argparse
import pathlib

def main():
    # Setup CLI Argument parsing
    parser = argparse.ArgumentParser(prog='stardist-inference', description='Segment images using Stardist pretrained models')

    # Define arguments
    parser.add_argument('--inputImages', dest='input_images', type=str, help='filepath to the directory containing the images', required=True)
    parser.add_argument('--pretrainedModel', dest='pretrained_model', type=str, required=True)
    parser.add_argument('--output', dest='output_folder', type=str, required=True)

    # Parse arguments
    args = parser.parse_args()
    input_images = args.input_images
    pretrained_model = args.pretrained_model
    output_folder = args.output_folder

    # Print parsed arguments
    print('*** Arguments:')
    print('inputImages = {}'.format(input_images))
    print('pretrained_model = {}'.format(pretrained_model))
    print('output = {}'.format(output_folder))
    
    # Load pretrained model
    if pretrained_model == '3D':
        model = StarDist3D.from_pretrained("3D_demo")
    else:
        model = StarDist2D.from_pretrained(pretrained_model)
    
    # Get list of input images
    images = listdir(input_images)
    
    # Create output dir if needed
    out = pathlib.Path(output_folder)
    out.mkdir(parents=True,exist_ok=True)
    
    # Threshold images
    for i in range(len(images)):
        image = images[i]
        print('*** Processing ' + image)
        # Open current image
        image_data = imread(join(input_images, image))
        # Predict labels
        labels, _ = model.predict_instances(normalize(image_data), axes='ZYX', n_tiles=(1,4,4))
        # Save result as 16bit tiled tiff
        imwrite(join(output_folder, image), np.uint16(image_data), tile=(1024,1024), compression='lzw', shape=image_data.shape, metadata={'axes': 'ZYX'})
        print('Done')
    print('*** Execution finished.')

if __name__ == "__main__":
    main()

