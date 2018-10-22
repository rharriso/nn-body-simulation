#include <thrust/device_vector.h>
#include <opencv2/opencv.hpp>

int main () {
  const int RGBA_IMAGE_SIZE = 16 * 3;
  // save image
  auto hostImage = thrust::host_vector<unsigned char>(RGBA_IMAGE_SIZE);
  thrust::fill(hostImage.begin(), hostImage.end(), 127);
  auto hostImage_ptr = thrust::raw_pointer_cast(hostImage.data());
  //hostImage = deviceImage; // copy device image to host
  cv::Mat imageMat(4,4, CV_8UC3);
  memcpy(imageMat.data, hostImage_ptr, sizeof(unsigned char) * RGBA_IMAGE_SIZE);
  
  cv::imwrite("/home/rharriso/Desktop/Test.png", imageMat);
}
