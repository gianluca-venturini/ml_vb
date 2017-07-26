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

# Check all the parameters of the training set generator
python generate_images.py --help
# Example of good dataset 50,000 samples 32x32
python generate_images.py --width 32 --height 32 --training text_ovelapping_model/training/good --images 100 --cuts 500 --horizontal_offset 2 --vertical_offset 2 --line_height 20 --monocrome_background=True
# Example of bug dataset 50,000 samples 32x32
python generate_images.py --width 32 --height 32 --training text_ovelapping_model/training/bug --images 100 --cuts 500 --vertical_offset 0 --line_height 4 --horizontal_offset -20 --skip_lines 2 --keep_lines 2 --monocrome_background True
python generate_images.py --width 32 --height 32 --training text_ovelapping_model/training/bug_horizontal --images 100 --cuts 500 --vertical_offset 0 --line_height 20 --horizontal_offset -60 --monocrome_background True --line_same_font --max_font_size 20

# Generate no text training dataset
python generate_images.py --training text_ovelapping_model/text_training/no_text --word 0 --model_name text --background_images images_no_text --images 100 --cuts 500 --monocrome_probability 0.1  --height 20 --width 20
# Generate text training dataset
python generate_images.py --training text_ovelapping_model/text_training/text --model_name text --background_images images --images 100 --cuts 500 --monocrome_probability 0.001 --vertical-offset 0 --monocrome_background True  --height 20 --width 20
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
# Train text / no text model
python keras_training.py --model_name text --num_good_train_images 10000 --num_bug_images 10000 --training text_training --dataset_good_name no_text --dataset_bug_name text --image_size 20
# Train the text overlap model
python keras_training.py --model_name text_overlap --num_good_train_images 30000 --num_bug_train_images 30000 --training training --dataset_good_name good --dataset_bug_name bug --image_size 32 --epochs 10 --text_overlap_architecture
```
The model will be saved in *keras_model* directory

# Test the results on a real image
```
cd text_ovelapping_model
python keras_verify_image.py --text_model text --text_overlap_treshold 0.5 --text_treshold 0.5 --text_step_size 32 --text_overlap_step_size 1
```
