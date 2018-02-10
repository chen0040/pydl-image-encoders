# pydl-image-encoders

Image encoders that convert varied-size images into fixed-length numpy arrays for machine learning

# Install

Git clone this project to your host computer. 

Make sure that your host computer or pyenv has installed the python packages specified in requirements.txt (it is 
recommended that you uses PyCharm IDE for python development as it will make this process easier for Python beginer) 

Copy the "pydl_image_encoders" folder from the git-cloned project into your  own project's root directory. Now you can use 
the the library in the same way as shown in the [pydl_image_encoders/demo](pydl_image_encoders/demo)


### Encode Varied-Size Image to Fix-Dimension Array: Inception Residual Network

The code below shows show to use GenSimWord2VecModel from [pydl_image_encoders/library/encoders/images/inception_resnet.py](pydl_image_encoders/library/encoders/images/inception_resnet.py)
to convert images of varied sizes into fixed-length array

The sample code can be found in the [pydl_image_encoders/demo/image_encoders/inception_resnet_encoder.py](pydl_image_encoders/demo/image_encoders/inception_resnet_encoder.py).

```python
from pydl_image_encoders.library.inception_resnet import InceptionResNetImageEncoder


def main():
    data_dir_path = '../very_large_data'
    img_dir_path = '../data/images'
    sample_images = [img_dir_path + '/dog.jpg', img_dir_path + '/cat.jpg']
    encoder = InceptionResNetImageEncoder()
    encoder.load_model(data_dir_path)
    for image_path in sample_images:
        label, class_id, predict_score = encoder.predict_image_file(image_path)
        print(encoder.encode_image_file(image_path).shape)
        print(encoder.encode_image_file(image_path, False).shape)
        print(label, class_id, predict_score)


if __name__ == '__main__':
    main()
```

By default the generated fixed-length array has uses the top layer of inception_resnet output, thus the resulted fixed-length 
array has 1001 dimension. However, if you want to uses the lower layer of the inception_reset as output, you can call 
encoder.encode_image_file(image_path, False) instead of encoder.encode_image_file(image_path) in the above 
sample code. this will output a fixed-length array of 1536 dimension. Not using the top layer of inception_resnet may be
quite suitable for problem that involves transfer learning with inception_resnet.
