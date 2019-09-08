
#ifndef _SGM_CUDA_H_
#define _SGM_CUDA_H_

// STL
#include <vector>

// OpenCV
#include <opencv2/core.hpp>

namespace stereo {

class SGMCuda {
 public:
  SGMCuda();

  ~SGMCuda();

  bool initialize(uint8_t p1, uint8_t p2);

  void generateDisparityMat(const cv::Mat &left, const cv::Mat &right,
                            cv::Mat &disp);

  double getSpentTimeForCur100Frames();

  bool end();

 private:
  static const int VEC_SIZE = 100;
  int spentTimeCnt;
  uint8_t p1, p2;
  std::vector<float> spentTime;
  bool finishCalled;
};
}

#endif