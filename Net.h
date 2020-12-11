#ifndef NET_H
#define NET_H

#include<torch/torch.h>


struct Net: torch::nn::Module
{
    Net(int num_features, int num_outputs)
    {
        neuron = register_module("neuron", torch::nn::Linear(num_features, num_outputs));        
    }
    
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.reshape({x.size(0), -1});
        x= neuron->forward(x);
        return x;
    }
    
    torch::nn::Linear neuron{nullptr};
};

#endif // NET_H
