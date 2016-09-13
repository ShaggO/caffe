#ifndef CAFFE_MEAN_LAYER_HPP_
#define CAFFE_MEAN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the mean operation of an array
 */
template <typename Dtype>
class MeanLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with AccuracyLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank @f$ k @f$ at which a prediction is considered
   *     correct.  For example, if @f$ k = 5 @f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  explicit MeanLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Mean"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }

  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (C) @f$
   *      the performance measure for each class to compute mean over
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      the computed mean performance measure: @f$
   *        \frac{1}{C} \sum\limits_{n=1}^C p_n
   *      @f$   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- MeanLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
