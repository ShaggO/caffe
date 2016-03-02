#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/meaniu_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MeanIULayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ =
    this->layer_param_.meaniu_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.meaniu_param().ignore_label();
  }
}

template <typename Dtype>
void MeanIULayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.meaniu_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // MeanIU is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  vector<int> top_shape_per_class(1);
  top_shape_per_class[0] = bottom[0]->shape(label_axis_);
  nums_buffer_.Reshape(top_shape_per_class);
  nums_tp_.Reshape(top_shape_per_class);
  nums_fp_.Reshape(top_shape_per_class);
}

template <typename Dtype>
void MeanIULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);

  Dtype outer_accum = 0;
  for (int i = 0; i < outer_num_; ++i) {
    caffe_set(nums_tp_.count(), Dtype(0), nums_tp_.mutable_cpu_data());
    caffe_set(nums_fp_.count(), Dtype(0), nums_fp_.mutable_cpu_data());
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      ++nums_buffer_.mutable_cpu_data()[label_value];
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      Dtype max_val = -1;
      Dtype cur_val;
      int max_id = 0;
      for (int k = 0; k < num_labels; ++k) {
        cur_val = bottom_data[i * dim + k * inner_num_ + j];
        if (cur_val > max_val) {
          max_val = cur_val;
          max_id  = k;
        }
      }
      if (max_id == label_value) {
        ++nums_tp_.mutable_cpu_data()[label_value];
      } else {
        ++nums_fp_.mutable_cpu_data()[max_id];
      }
    }
    Dtype inner_accum = 0;
    Dtype num_inner_labels = 0;
    for (int j = 0; j < num_labels; ++j) {
      if (has_ignore_label_ && j == ignore_label_) {
        continue;
      }
      if (nums_buffer_.mutable_cpu_data()[j] > 0) {
        ++num_inner_labels;
        inner_accum += nums_tp_.mutable_cpu_data()[j] /
            (nums_buffer_.mutable_cpu_data()[j] + nums_fp_.mutable_cpu_data()[j]);
      }
    }
    inner_accum /= num_inner_labels;
    outer_accum += inner_accum;
  }

  top[0]->mutable_cpu_data()[0] = outer_accum / outer_num_;
  // MeanIU layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MeanIULayer);
REGISTER_LAYER_CLASS(MeanIU);

}  // namespace caffe
