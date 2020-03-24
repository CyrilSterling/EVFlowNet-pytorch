import os
from tqdm import trange
from tqdm import tqdm
import numpy as np
from datetime import datetime
from losses import *

import torch

from config import configs
from data_loader import EventData
from EVFlowNet import EVFlowNet

def main():
    args = configs()

    if args.training_instance:
        args.load_path = os.path.join(args.load_path, args.training_instance)
    else:
        args.load_path = os.path.join(args.load_path,
                                      "evflownet_{}".format(datetime.now()
                                                            .strftime("%m%d_%H%M%S")))
    if not os.path.exists(args.load_path):
        os.makedirs(args.load_path)

    EventDataset = EventData(args.data_path, 'train')
    EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)

    # model
    EVFlowNet_model = EVFlowNet(args).cuda()

    #para = np.load('D://p.npy', allow_pickle=True).item()
    #EVFlowNet_model.load_state_dict(para)
    EVFlowNet_model.load_state_dict(torch.load(args.load_path+'/../model'))

    #EVFlowNet_parallelmodel = torch.nn.DataParallel(EVFlowNet_model)
    # optimizer
    optimizer = torch.optim.Adam(EVFlowNet_model.parameters(), lr=args.initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.learning_rate_decay)
    loss_fun = TotalLoss(args.smoothness_weight)

    iteration = 0
    size = 0
    EVFlowNet_model.train()
    for epoch in range(100):
        loss_sum = 0.0
        print('*****************************************')
        print('epoch:'+str(epoch))
        for event_image, prev_image, next_image, _ in tqdm(EventDataLoader):
            event_image = event_image.cuda()
            prev_image = prev_image.cuda()
            next_image = next_image.cuda()

            optimizer.zero_grad()
            flow_dict = EVFlowNet_model(event_image)

            loss = loss_fun(flow_dict, prev_image, next_image, EVFlowNet_model)
            
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            iteration += 1
            size += 1
            print(loss)
            if iteration % 100 == 0:
                print('iteration:', iteration)
                print('loss:', loss_sum/100)
                loss_sum = 0.0
            torch.save(EVFlowNet_model.state_dict(), args.load_path+'/model%d'%epoch)
        if epoch % 4 == 3:
            scheduler.step()
        print('iteration:', iteration)
        print('loss:', loss_sum/size)
        size = 0


    

if __name__ == "__main__":
    main()
