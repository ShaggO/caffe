#include <cfloat>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/meaniu_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class MeanIULayerTest : public CPUDeviceTest<Dtype> {
 protected:
  MeanIULayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    vector<int> shape(2);
    shape[0] = 100;
    shape[1] = 10;
    blob_bottom_data_->Reshape(shape);
    shape.resize(1);
    blob_bottom_label_->Reshape(shape);
    FillBottoms();

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual void FillBottoms() {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng =
          static_cast<caffe::rng_t*>(rng->generator());
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      label_data[i] = (*prefetch_rng)() % 10;
    }
  }

  virtual ~MeanIULayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MeanIULayerTest, TestDtypes);

TYPED_TEST(MeanIULayerTest, TestSetup) {
  LayerParameter layer_param;
  MeanIULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}
/*
TYPED_TEST(MeanIULayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  MeanIULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++num_correct_labels;
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, 1e-4);
}
*/
TYPED_TEST(MeanIULayerTest, TestForwardWithSpatialAxes) {
  this->blob_bottom_data_->Reshape(2, 10, 4, 5);
  vector<int> label_shape(3);
  label_shape[0] = 2; label_shape[1] = 4; label_shape[2] = 5;
  this->blob_bottom_label_->Reshape(label_shape);
  this->FillBottoms();
  LayerParameter layer_param;
  layer_param.mutable_meaniu_param()->set_axis(1);
  MeanIULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  vector<int> label_offset(3);
  TypeParam sum_IU = 0;
  for (int n = 0; n < this->blob_bottom_data_->num(); ++n) {
    vector<TypeParam> enumerator(this->blob_bottom_data_->channels(),0);
    vector<TypeParam> divisor(this->blob_bottom_data_->channels(),0);
    vector<bool> label_occured(this->blob_bottom_data_->channels(),false);
    for (int h = 0; h < this->blob_bottom_data_->height(); ++h) {
      for (int w = 0; w < this->blob_bottom_data_->width(); ++w) {
        label_offset[0] = n; label_offset[1] = h; label_offset[2] = w;
        const int correct_label =
            static_cast<int>(this->blob_bottom_label_->data_at(label_offset));
        label_occured[correct_label] = true;
        ++divisor[correct_label]; // Add one to t_i where i is the class
        max_value = -FLT_MAX;
        max_id = 0;
        for (int c = 0; c < this->blob_bottom_data_->channels(); ++c) {
          const TypeParam pred_value =
              this->blob_bottom_data_->data_at(n, c, h, w);
          if (pred_value > max_value) {
            max_value = pred_value;
            max_id = c;
          }
        }
        if (max_id == correct_label) {
          ++enumerator[correct_label]; // Add one to n_ii where i is the class
        } else {
          ++divisor[max_id]; // Add one to n_ji where i is the class and j is the prediction
        }
      }
    }
    TypeParam n_IU = 0;
    TypeParam n_labels = 0;
    for (int i = 0; i < label_occured.size(); ++i) {
      if (label_occured[i]) {
        ++n_labels;
        n_IU += enumerator[i] / divisor[i];
      }
    }
    n_IU = n_IU / n_labels;
    sum_IU += n_IU;
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              sum_IU / TypeParam(this->blob_bottom_label_->num()), 1e-4);
}

TYPED_TEST(MeanIULayerTest, TestForwardWithSpatialAxesIgnoreLabel) {
  LayerParameter layer_param;
  const TypeParam kIgnoreLabelValue = -1;
  layer_param.mutable_meaniu_param()->set_ignore_label(kIgnoreLabelValue);
  // Manually set some labels to the ignore label value (-1).
  this->blob_bottom_label_->mutable_cpu_data()[2] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[5] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[32] = kIgnoreLabelValue;
  this->blob_bottom_data_->Reshape(2, 10, 4, 5);
  vector<int> label_shape(3);
  label_shape[0] = 2; label_shape[1] = 4; label_shape[2] = 5;
  this->blob_bottom_label_->Reshape(label_shape);
  this->FillBottoms();
  layer_param.mutable_meaniu_param()->set_axis(1);
  MeanIULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  vector<int> label_offset(3);
  TypeParam sum_IU = 0;
  for (int n = 0; n < this->blob_bottom_data_->num(); ++n) {
    vector<TypeParam> enumerator(this->blob_bottom_data_->channels(),0);
    vector<TypeParam> divisor(this->blob_bottom_data_->channels(),0);
    vector<bool> label_occured(this->blob_bottom_data_->channels(),false);
    for (int h = 0; h < this->blob_bottom_data_->height(); ++h) {
      for (int w = 0; w < this->blob_bottom_data_->width(); ++w) {
        label_offset[0] = n; label_offset[1] = h; label_offset[2] = w;
        const int correct_label =
            static_cast<int>(this->blob_bottom_label_->data_at(label_offset));
        if (correct_label == kIgnoreLabelValue) {
            continue;
        }
        label_occured[correct_label] = true;
        ++divisor[correct_label]; // Add one to t_i where i is the class
        max_value = -FLT_MAX;
        max_id = 0;
        for (int c = 0; c < this->blob_bottom_data_->channels(); ++c) {
          const TypeParam pred_value =
              this->blob_bottom_data_->data_at(n, c, h, w);
          if (pred_value > max_value) {
            max_value = pred_value;
            max_id = c;
          }
        }
        if (max_id == correct_label) {
          ++enumerator[correct_label]; // Add one to n_ii where i is the class
        } else {
          ++divisor[max_id]; // Add one to n_ji where i is the class and j is the prediction
        }
      }
    }
    TypeParam n_IU = 0;
    TypeParam n_labels = 0;
    for (int i = 0; i < label_occured.size(); ++i) {
      if (label_occured[i]) {
        ++n_labels;
        n_IU += enumerator[i] / divisor[i];
      }
    }
    n_IU = n_IU / n_labels;
    sum_IU += n_IU;
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              sum_IU / TypeParam(this->blob_bottom_label_->num()), 1e-4);
}

}  // namespace caffe
