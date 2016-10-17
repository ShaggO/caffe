#ifndef CAFFE_INTERSECTION_OVER_UNION_LAYER_HPP_
#define CAFFE_INTERSECTION_OVER_UNION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification intersection over union for a one-of-many
 *        classification task.
 */
template <typename Dtype>
class IntersectionOverUnionLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides IntersectionOverUnionParameter intersectionoverunion_param,
   *     with AccuracyLayer options:
   *   - axis (\b optional, default 1).
   *   - has_ignore_label (\b optional, default false)
   *   - ignore_label (\b optional, default 0)
   */
  explicit IntersectionOverUnionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InterSectionOverUnion"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // If there are two top blobs, then the second blob will contain
  // intersection over union per class.
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlos() const { return 2; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
   *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
   *      label @f$ \hat{l}_n @f$ given by its maximal index:
   *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels @f$ l @f$, an integer-valued Blob with values
   *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
   *      indicating the correct class label among the @f$ K @f$ classes
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      the computed mean of intersection over union: @f$
   *        \frac{1}{n_{cl} \sum_i \frac{n_{ii}}{t_i+\sum_j n_{ji} - n_{ii}}
   *      @f$
   *   -# @f$ (C) @f$
   *      intersection over union per class
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- IntersectionOverUnionLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  int label_axis_, outer_num_, inner_num_;

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// Keeps count of outer number with occurences of each label
  Blob<Dtype> nums_buffer_;

  /// Keeps counts of the number of true predictions (intersection)
  /// for each class.
  Blob<Dtype> nums_intersect_;
  /// Keeps counts of the number of predictions and ground truths (union)
  /// for each class.
  Blob<Dtype> nums_union_;
};

}  // namespace caffe

#endif  // CAFFE_INTERSECTION_OVER_UNION_LAYER_HPP_
