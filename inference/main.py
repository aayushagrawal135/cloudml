from __future__ import print_function
from flask import Flask, json, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

app = Flask(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def preprocess(image):
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def infer(image):
    model = Net()
    model.load_state_dict(torch.load('/mnt/model.pth'))
    model.eval()
    output = model(image)
    result = output.squeeze().argmax().item()
    return result


@app.route('/')
def home():
    return "Do a curl request like this: curl -X POST -F image=@img_1.jpg 'http://127.0.0.1:5000/infer'"


@app.route('/infer', methods=['POST'])
# @app.route('/infer')
def get_result():
    res = {}
    file = request.files['image']
    if not file:
        res['status'] = 'missing image'
    else:
        res['status'] = 'success'
        image = Image.open(file.stream)
        output = infer(preprocess(image))
        res['result'] = output

    return json.dumps(res)


app.run(host='0.0.0.0')
