import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import CustomModel
from dataset import CustomDataset
import os
import torch
import pandas as pd

if __name__ == '__main__':


    opt = config.parse_opt()
    test = pd.read_csv(os.path.join(opt.DATA_PATH, opt.TEST_DATA))


    test_dataset = CustomDataset(data=test, root=f'./data/cifar_10/{opt.FOLDER}/')
    test_dataloader = DataLoader(test_dataset, batch_size=opt.TEST_BATCH_SIZE,
                                 num_workers=0,
                                 shuffle=False,
                                 #         pin_memory=True
                                 )

    model = CustomModel(opt)

    model.cuda()

    filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED),'checkpoint/{}.pth'.format(opt.EPOCH - 1))
    # filename = os.path.join(opt.OUTPUT, 'resnet100', str(opt.PERCENTAGE), str(opt.CHECKPOINT),'checkpoint/{}.pth'.format(opt.EPOCH - 1))
    model.load_state_dict(torch.load(filename))

    model.eval()

    predictions_list = []
    for idx, batch in enumerate(tqdm(test_dataloader)):
        cifar_idxes, labels, features = batch

        features = features.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            outputs = model(features)

            predictions = outputs.detach().cpu().numpy()
            predictions_list.append(predictions)

    predictions = np.vstack(predictions_list)
    predictions = np.argmax(predictions, axis=1)
    test['prediction'] = predictions

    report = classification_report(test['label'], test['prediction'], digits=4, output_dict=True)
    #print(report)
    report_df = pd.DataFrame(report).transpose()

    #filename = os.path.join(opt.OUTPUT, opt.LAYER1, str(opt.PERCENTAGE), str(opt.SEED), 'report/{}_{}.csv'.format(i, 'test'))
    filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED),
                            'report/{}_{}.csv'.format(9, opt.FOLDER))
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    report_df.to_csv(filename)
        ####