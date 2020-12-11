#ifndef CONFIG_H
#define CONFIG_H

#include<torch/torch.h>

namespace Config
{
    constexpr size_t trainBatchSize = 4;
    constexpr size_t testBatchSize = 100;
    constexpr size_t epochs = 1000;
    constexpr size_t logInterval = 20;
    constexpr char datasetPath[] = "C:/study/test/libtorch/build-linear-Desktop_Qt_5_14_1_MSVC2017_64bit-Debug/data/BostonHousing.csv";
    constexpr torch::DeviceType device = torch::kCUDA;
}

#endif // CONFIG_H
