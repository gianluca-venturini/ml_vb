# Steps for initializing the repository

Assuming that you already have:
- pip
- node.js + npm
```
./init.sh
```

# In order to test the training set generator
```
# Activate the virtualenv
source bin/activate
python generate_images.py --web_server=True

# Example of good dataset 50,000 samples 64x64
python generate_images.py --width 64 --height 64 --training training/good --words 0 --images 100 --cuts 500
# Example of bug dataset 50,000 samples 64x64
python generate_images.py --width 64 --height 64 --training training/bug --images 100 --cuts 500
```

# Check all the parameters of the training set generator
```
python generate_images.py --help
```

# Work with the image generator
The image generator is a small typescript web app. You will need to compile it with `gulp`. All the needed dependencies are installed by `init.sh`.
```
cd image_generator
[make your changes]
gulp
```
if you use any other dependency remember to `npm install <dependency> --save-dev`

# Train the model
```
cd text_ovelapping_model
python keras_training.py
```
The model will be saved in *keras_model* directory

# Test the results on a real image
```
cd text_ovelapping_model
python keras_verify_image.py
```
