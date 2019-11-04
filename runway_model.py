# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =========================================================================

# This example contains the minimum specifications and requirements
# to port a machine learning model to Runway.

# For more instructions on how to port a model to Runway, see the Runway Model
# SDK docs at https://sdk.runwayml.com

# RUNWAY
# www.runwayml.com
# hello@runwayml.com

# =========================================================================

# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import text, image
from keras_model import KerasModel

# Setup the model, initialize weights, set the configs of the model, etc.
# Every model will have a different set of configurations and requirements.
# Check https://docs.runwayapp.ai/#/python-sdk to see a complete list of
# supported configs. The setup function should return the model ready to be
# used.
setup_options = {
    'model_checkpoint': runway.file(extension='.h5'),
}

@runway.setup(options=setup_options)
def setup(opts):
    msg = '[SETUP] Loading model: {}'
    print(msg.format(opts['model_checkpoint']))
    model = KerasModel(opts)
    return model

# Every model needs to have at least one command. Every command allows to send
# inputs and process outputs. To see a complete list of supported inputs and
# outputs data types: https://sdk.runwayml.com/en/latest/data_types.html
@runway.command(name='classify',
                inputs={ 'image': image() },
                outputs={ 'text': text() },
                description='Predict cats or dogs in photo')
def classify(model, args):
    print('[CLASSIFY] Classifying image')
    predictions = model.classify(args['image'])
    return {
        'text': predictions
    }

if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run()
    # runway.run(model_options={'model_checkpoint': 'checkpoints/cats_and_dogs_small_2.h5'})


## Now that the model is running, open a new terminal and give it a command to
## classify an image. 

# Fisrt encode the image
# IMAGE64="data:image/jpeg;base64,"$(base64 -w 0 -i dog.jpg)

# Then call the API
# curl \
#   -H "content-type: application/json" \
#   -d "{ \"image\": \"$IMAGE64\" }"
#   localhost:9000/classify
