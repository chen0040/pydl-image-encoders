from pydl_image_encoders.library.inception_resnet import InceptionResNetImageEncoder


def main():
    data_dir_path = '../very_large_data'
    img_dir_path = '../data/images'
    sample_images = [img_dir_path + '/dog.jpg', img_dir_path + '/cat.jpg']
    encoder = InceptionResNetImageEncoder()
    encoder.load_model(data_dir_path)
    for image_path in sample_images:
        label, class_id, predict_score = encoder.predict_image_file(image_path)
        print(encoder.encode_image_file(image_path, True).shape)
        print(encoder.encode_image_file(image_path, False).shape)
        print(label, class_id, predict_score)


if __name__ == '__main__':
    main()