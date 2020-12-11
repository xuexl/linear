#ifndef TRAIN_H
#define TRAIN_H

#include"Net.h"
#include"Config.h"

#include<memory>

#include"IO.h"

template<typename DataLoader>
void train(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer, size_t epoch, size_t data_size)
{
    network->train();
    
    for(auto& batch: loader)
    {
        auto data = batch.data.to(Config::device);
        auto target = batch.target.to(Config::device).view({-1});
        
        auto output = network->forward(data);
        
        auto loss = torch::mse_loss(output, target);
        cout("****loss*****");
        cout(loss);
        cout("****loss*****");
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
                
//        float Loss = .0f;
//        Loss += loss.template item<float>();

//        size_t index = 0;
//        if(++index % Config::logInterval == 0)
//        {
//            auto end = std::min(data_size, (index+1)*Config::trainBatchSize);
//            std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
//                           << "\t\tLoss: " << Loss / end << std::endl;
//        }
    }

}

#endif // TRAIN_H
