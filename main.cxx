#include"IO.h"

#include"Net.h"
#include"Data.h"
#include"Train.h"
#include"Test.h"
#include"Config.h"

int main()
{

    torch::manual_seed(1);   
    
    auto data = readInfo();
    
    auto train_set = CustomDataset(data.first).map(torch::data::transforms::Stack<>());
    auto train_size = train_set.size().value();
    
    auto train_loader = torch::data::make_data_loader(std::move(train_set), Config::trainBatchSize);
    
    ///
    auto test_set = CustomDataset(data.second).map(torch::data::transforms::Stack<>());
    auto test_size = test_set.size().value();
    
    auto test_loader = torch::data::make_data_loader(std::move(test_set), Config::testBatchSize);
    
    auto net = std::make_shared<Net>(13, 1);
    net->to(Config::device);
    
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.000001));
    
    for(size_t i = 0; i < Config::epochs; ++i)
    {
        train(net, *train_loader, optimizer, i + 1, train_size);
        
        if(i == (Config::epochs - 1))
        {
            test(net, *test_loader, test_size);            
        }
    }
    
    return EXIT_SUCCESS;
}
