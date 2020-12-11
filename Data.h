#ifndef DATA_H
#define DATA_H

#include "CSVReader.h"
#include"Config.h"

#include<torch/torch.h>


std::vector<std::vector<float>> normalize_feature(std::vector<std::vector<std::string> > feat, int rows, int cols) 
{
  std::vector<float> input(cols, 1);    //初始化data
  std::vector<std::vector<float>> data(rows, input);

  for (int i = 0; i < cols; i++) 
  {   // each column has one feature
    // initialize the maximum element with 0 
    // std::stof is used to convert string to float
    float maxm = std::stof(feat[1][i]);
    float minm = std::stof(feat[1][i]);

    // Run the inner loop over rows (all values of the feature) for given column (feature) 
    for (int j = 1; j < rows; j++) 
    {
      // check if any element is greater  
      // than the maximum element 
      // of the column and replace it 
      if (std::stof(feat[j][i]) > maxm)
        maxm = std::stof(feat[j][i]);

      if (std::stof(feat[j][i]) < minm)
        minm = std::stof(feat[j][i]);
    }

    // From above loop, we have min and max value of the feature
    // Will use min and max value to normalize values of the feature
    for (int j = 0; j < rows-1; j++) 
    {
      // Normalize the feature values to lie between 0 and 1
      data[j][i] = (std::stof(feat[j+1][i]) - minm)/(maxm - minm);
    }
  }

  return data;
}



using Data = std::vector<std::pair<std::vector<float>, float>>;


class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{
    using Example = torch::data::Example<>;
        
public:
    CustomDataset(const Data& data):data(data)
    {        
    }
    
    Example get(size_t index)
    {
        int fSize = data[index].first.size();
        auto tdata = torch::from_blob(&data[index].first, {fSize, 1});
        auto toutput = torch::from_blob(&data[index].second, {1});
        return {tdata, toutput};
    }
    
    torch::optional<size_t> size() const 
    {
        return data.size();
    }
    
private:
    Data data;
    
};


std::pair<Data, Data> readInfo() 
{
    Data train, test;
    
    CSVReader reader(Config::datasetPath);
    std::vector<std::vector<std::string> > dataList = reader.getData();
    
    
    int N = dataList.size();      // Total number of data points
    // As last column is the output, feature size will be number of column minus one.
    int fSize = dataList[0].size() - 1;

    int limit = 0.8*N;    // 80 percent data for training and rest 20 percent for validation
    
    std::vector<float> input(fSize, 1);
    std::vector<std::vector<float>> data(N, input);
    
    // Normalize data
    data = normalize_feature(dataList, N, fSize);
        
    for (int i=1; i < N; i++) 
    {
        for (int j= 0; j < fSize; j++) 
        {
            input[j] = data[i-1][j];
        }
        
        float output = std::stof(dataList[i][fSize]);
        
        // Split data data into train and test set
        if (i <= limit) 
        {
            train.push_back({input, output});
        } 
        else 
        {
            test.push_back({input, output});
        }
    }

    // Shuffle training data
    std::random_shuffle(train.begin(), train.end());
    
    return std::make_pair(train, test);
}

#endif // DATA_H
