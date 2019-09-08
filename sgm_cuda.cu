

// cuda
#include <cuda.h>
//#include <cuda_runtime.h>

// STL
#include <numeric>

// sgm
#include <disparity_method.h>

// Original
#include <sgm_cuda.h>

namespace stereo {

SGMCuda::SGMCuda() : p1(100), p2(100), finishCalled(false) {}

SGMCuda::~SGMCuda() {
  if (!finishCalled) {
    finish_disparity_method();
    finishCalled = true;
  }
}

bool SGMCuda::initialize(uint8_t p1, uint8_t p2) {
  init_disparity_method(p1, p2);
  finishCalled = false;
  return true;
}

void SGMCuda::generateDisparityMat(const cv::Mat &leftSrc,
                                   const cv::Mat &rightSrc, cv::Mat &disp) {
  float elapsedTime;

  cv::Mat left = leftSrc.clone();
  cv::Mat right = rightSrc.clone();

  if (1 < left.channels()) {
    cv::cvtColor(left, left, CV_RGB2GRAY);
  }

  if (1 < right.channels()) {
    cv::cvtColor(right, right, CV_RGB2GRAY);
  }

  cv::Mat dispMat = compute_disparity_method(left, right, &elapsedTime, "", "");

  {
    static int cnt = 0;
    spentTime.push_back(elapsedTime);
    cnt++;

    if (VEC_SIZE <= cnt) {
      cnt = 0;
    }
  }

  dispMat.copyTo(disp);
}

bool SGMCuda::end() {
  if (!finishCalled) {
    finish_disparity_method();
    finishCalled = true;
  }
  return true;
}

double SGMCuda::getSpentTimeForCur100Frames() {
  return std::accumulate(spentTime.begin(), spentTime.end(), 0.0) /
         (double)(spentTime.size());
}
}