#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>

const int PIXEL_RGBA_RATIO = 3;

/**
 * Struct representing a body 
 */
struct Body {
  float x, y, vx, vy;

  __host__ __device__
    Body (const Body &otherBody):
      x(otherBody.x), y(otherBody.y), vx(otherBody.vx), vy(otherBody.vy) {};
  __host__ __device__
    Body (float _x, float _y, float _vx, float _vy): x(_x), y(_y), vx(_vx), vy(_vy) {};
  __host__ __device__
    Body () {};
};

/**
 * Image Coordinates
 */
struct ImageCoord {
  int x, y, imageDim;

  __host__ __device__
    // create coordinate from components
    ImageCoord (int _x, int _y, int _imageDim): x(_x), y(_y), imageDim(_imageDim) {};

  __host__ __device__
    // default init
    ImageCoord (): x(0), y(0), imageDim(0) {};

  __host__ __device__
    // create image coordinate from a body's position
    ImageCoord (const Body &body, int _imageDim):  imageDim(_imageDim) {
      auto halfImageDim = imageDim / 2;
      x = (body.x + 1.) * halfImageDim;
      y = (body.y + 1.) * halfImageDim;
    };

  __host__ __device__
    int toOffset() {
      return x + y * imageDim;
    };
};

/**
 * Program to initialize bodies with random values between -1,1
 */
struct initRandomPrg
{
  float minValue, maxValue;
  int seed;

  __host__ __device__
    initRandomPrg(int _seed=0, float _mnV=-1.f, float _mxV=1.f):
      seed(_seed), minValue(_mnV), maxValue(_mxV) {};

  __host__ __device__
    Body operator()(const unsigned int idx) const
    {
      thrust::default_random_engine rng(seed);
      thrust::uniform_real_distribution<float> dist(minValue, maxValue);
      rng.discard(idx);

      return Body{
        dist(rng), dist(rng),
          dist(rng), dist(rng)
      };
    }
};


struct mapBodyToPixelCounts
{
  int *pixelCounts;
  const int imageDim;

  __host__ __device__
    mapBodyToPixelCounts(
        int _imageDim,
        int *_pixelCounts
        ): imageDim(_imageDim), pixelCounts(_pixelCounts) {};

  __device__
    void operator()(const Body &body) const
    {
      auto offset = ImageCoord(body, imageDim).toOffset();
      atomicAdd(&pixelCounts[offset], 1);
    }
};

struct mapPixelCountToRGBA
{
  const int BODY_COUNT_GRAYSCALE_RATIO = 1; // like 12 points in white
  const int *pixelCount_ptr;
  unsigned char *image_ptr;
  __host__ __device__
  mapPixelCountToRGBA(
    int *_pixelCounts,
    unsigned char *_image
  ): pixelCount_ptr(_pixelCounts), image_ptr(_image) {};

  __host__ __device__
  void operator()(const unsigned int idx) {
    auto count = pixelCount_ptr[idx];
    auto grayValue = min(BODY_COUNT_GRAYSCALE_RATIO * count, 127);
    // assign rgba values
    auto baseIdx = idx * PIXEL_RGBA_RATIO;
    image_ptr[baseIdx] = grayValue;
    image_ptr[baseIdx + 1] = grayValue;
    image_ptr[baseIdx + 2] = grayValue;
    //image_ptr[baseIdx + 3] = 0;
  }
};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: n-body [output file]" << std::endl;
    return 1;
  }


  int const BODY_COUNT = 10e6;
  int const IMAGE_DIM = 400;
  int const PIXEL_COUNT = IMAGE_DIM * IMAGE_DIM;
  int const RGBA_IMAGE_SIZE = PIXEL_COUNT * PIXEL_RGBA_RATIO; // image has 4 values per pixels


  // initilize bodies
  auto bodies = thrust::device_vector<Body>(BODY_COUNT);
  auto index_sequence_begin = thrust::counting_iterator<unsigned int>(0);

  thrust::transform(
      index_sequence_begin,
      index_sequence_begin + BODY_COUNT,
      bodies.begin(),
      initRandomPrg(std::time(0))
      );

  std::cout << "Initialized Bodies" << "\n\n";

  // initialize pixel counts 
  auto pixelCounts = thrust::device_vector<int>(PIXEL_COUNT);
  auto pixelCount_ptr = thrust::raw_pointer_cast(pixelCounts.data());
  thrust::fill(pixelCounts.begin(), pixelCounts.end(), 0);

  std::cout << "Initialized Pxels" << "\n\n";

  // get pixel counts
  thrust::for_each(bodies.begin(), bodies.end(), mapBodyToPixelCounts(IMAGE_DIM, pixelCount_ptr));

  std::cout << "Mapped Points to Pixels" << "\n\n";

  auto deviceImage = thrust::device_vector<unsigned char>(RGBA_IMAGE_SIZE);
  auto deviceImage_ptr = thrust::raw_pointer_cast(deviceImage.data());
  
  thrust::for_each(
    index_sequence_begin,
    index_sequence_begin + PIXEL_COUNT,
    mapPixelCountToRGBA(pixelCount_ptr, deviceImage_ptr)
  ); 

  std::cout << "Initialized Image Pixels" << "\n\n";

  for (auto i = 0; i < 10; i++) {
    Body b = bodies[i];
    auto bc = ImageCoord(b, IMAGE_DIM);
    auto offset = ImageCoord(b, IMAGE_DIM).toOffset();
    auto baseIdx = i * PIXEL_RGBA_RATIO;

    auto red = (int)deviceImage[baseIdx]; 
    auto blue = (int)deviceImage[baseIdx + 1]; 
    auto green = (int)deviceImage[baseIdx + 2]; 
    // auto alpha = (int)deviceImage[baseIdx + 3]; 

    std::cout << b.x << '\t' << b.y << '\t';
    std::cout << offset << ' ' <<  bc.x << ',' << bc.y << "\t\t";
    std::cout << red << ", " <<  blue << ", " << green;
    std::cout << '\n';
  }

  auto max = thrust::reduce(
      pixelCounts.begin(), pixelCounts.end(),
      0, thrust::maximum<int>()
  );

  std::cout << "Max Pixel Count: \t" << max << '\n';
  std::cout << "Body Count: \t\t" << BODY_COUNT << '\n';

  // save image
  auto hostImage = thrust::host_vector<unsigned char>(RGBA_IMAGE_SIZE);
  // thrust::fill(hostImage.begin(), hostImage.end(), 127);
  auto hostImage_ptr = thrust::raw_pointer_cast(hostImage.data());
  //thrust::copy(hostImage.begin(), hostImage.end(), deviceImage.begin());
  cudaDeviceSynchronize();
  cudaMemcpy(hostImage_ptr, deviceImage_ptr, RGBA_IMAGE_SIZE, cudaMemcpyDeviceToHost);

  cv::Mat imageMat(IMAGE_DIM,IMAGE_DIM, CV_8UC3);
  memcpy(imageMat.data, hostImage_ptr, sizeof(unsigned char) * RGBA_IMAGE_SIZE);

  //cv::imwrite("/home/rharriso/Desktop/Test.png", imageMat);
  cv::imwrite(argv[1], imageMat);
  return 0;
}
