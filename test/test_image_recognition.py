import json
import sys
import os
sys.path.append('../')

from entrobus import image_recognition

if __name__ == "__main__":
    config = {
        'phase': 'train',
        'train_dir': r"D:\LJS\Documents\competition\TransferLearning\hymenoptera_data\train",
        'gpu': False,
        'epochs': 1,
        'batch_size': 4
    }
    config = json.dumps(config)
    train_rlt = image_recognition.train(config)
    train_rlt = json.loads(train_rlt)

    model_path = train_rlt['model_path']
    test_config = {
        'phase': 'test',
        'url': 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1540453724136&di=599388f0d7570a59629bcb1acc01588e&imgtype=0&src=http%3A%2F%2Fphotocdn.sohu.com%2F20150120%2Fmp700680_1421752454518_9.jpeg',
        'model_path': model_path
    }
    test_config = json.dumps(test_config)

    test_rlt = image_recognition.test(test_config)
    print(json.dumps(test_rlt))