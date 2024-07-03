import os
import json

import torch
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from model import swin_base_patch4_window7_224 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.143)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=5).to(device)
    # load model weights
    model_weight_path = "./weights/model-86.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    # load image
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    all_dir = os.path.join(data_root, "data_set")  # flower data set path
    # img_path_list = ["../tulip.jpg", "../rose.jpg"]
    img_list = []
    test_dir = os.path.join(all_dir, "jpg")  # test
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transform)
    for img_path, idx in test_datasets.imgs:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        # img_path = "./tulip.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)


        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)

            predict_cla = torch.argmax(predict).numpy()

        print_res = "image: {}  class: {}   prob: {:.3}".format(img_path, class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        print(print_res)
        # plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                               predict[i].numpy()))
        # plt.show()


if __name__ == '__main__':
    main()
