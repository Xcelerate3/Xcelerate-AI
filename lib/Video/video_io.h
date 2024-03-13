#ifndef xcelerate_VIDEO_VIDEO_IO_H_
#define xcelerate_VIDEO_VIDEO_IO_H_

#include <xcelerate/core/common.h>
#include <xcelerate/video/optical_flow.h>
#include <xcelerate/video/video_decoder.h>
#include <opencv2/opencv.hpp>
#include <random>

#include <istream>
#include <ostream>

namespace xcelerate {

TORCH_API void ClipTransformRGB(
    const unsigned char* buffer_rgb,
    const int crop_size,
    const int length_rgb,
    const int channels_rgb,
    const int sampling_rate_rgb,
    const int height,
    const int width,
    const int h_off,
    const int w_off,
    const bool mirror_me,
    const std::vector<float>& mean_rgb,
    const std::vector<float>& inv_std_rgb,
    float* transformed_clip);

TORCH_API void ClipTransformOpticalFlow(
    const unsigned char* buffer_rgb,
    const int crop_size,
    const int length_of,
    const int channels_of,
    const int sampling_rate_of,
    const int height,
    const int width,
    const cv::Rect& rect,
    const int channels_rgb,
    const bool mirror_me,
    const int flow_alg_type,
    const int flow_data_type,
    const int frame_gap_of,
    const bool do_flow_aggregation,
    const std::vector<float>& mean_of,
    const std::vector<float>& inv_std_of,
    float* transformed_clip);

} // namespace xcelerate

#endif // xcelerate_VIDEO_VIDEO_IO_H_
