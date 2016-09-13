#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/mean_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MeanLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MeanLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape(0);  // Mean performance measure is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MeanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype measure = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int num_measures = bottom[0]->shape(0); // For now just have one axis

  int count = 0;
  for (int i = 0; i < num_measures; ++i) {
    measure += bottom_data[i];
    ++count;
  }

  // LOG(INFO) << "Mean measure: " << accuracy;
  top[0]->mutable_cpu_data()[0] = measure / count;
  // Mean layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MeanLayer);
REGISTER_LAYER_CLASS(Mean);

}  // namespace caffe
