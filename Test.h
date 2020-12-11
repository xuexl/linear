#ifndef TEST_H
#define TEST_H

#include"Net.h"
#include"Config.h"

#include<memory>

template <typename DataLoader>
void test(std::shared_ptr<Net> network, DataLoader& loader, size_t data_size)
{
    network->eval();
    
    for(const auto& batch: loader)
    {
        auto data = batch.data.to(Config::device);
        auto targets = batch.target.to(Config::device).view({-1});
                
        auto output = network->forward(data);
        
        std::cout << "Predicted:"<< output[0].template item<float>() << "\t" << "Groundtruth: "
                  << targets[1].template item<float>() << std::endl;
        std::cout << "Predicted:"<< output[1].template item<float>() << "\t" << "Groundtruth: "
                  << targets[1].template item<float>() << std::endl;
        std::cout << "Predicted:"<< output[2].template item<float>() << "\t" << "Groundtruth: "
                  << targets[2].template item<float>() << std::endl;
        std::cout << "Predicted:"<< output[3].template item<float>() << "\t" << "Groundtruth: "
                  << targets[3].template item<float>() << std::endl;
        std::cout << "Predicted:"<< output[4].template item<float>() << "\t" << "Groundtruth: "
                  << targets[4].template item<float>() << std::endl;
            
//        auto loss = torch::mse_loss(output, target);
//        break;
    }
    
}

#endif // TEST_H
