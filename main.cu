#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>

const int PIXEL_RGBA_RATIO = 4;

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

  __host__ __device__
    initRandomPrg(float _mnV=-1.f, float _mxV=1.f): minValue(_mnV), maxValue(_mxV) {};

  __host__ __device__
    Body operator()(const unsigned int idx) const
    {
      thrust::default_random_engine rng;
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
  const int BODY_COUNT_GRAYSCALE_RATIO = 10; // like 12 points in white
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
    auto grayValue = min(BODY_COUNT_GRAYSCALE_RATIO * count, 255);
    // assign rgba values
    auto baseIdx = idx * PIXEL_RGBA_RATIO;
    image_ptr[baseIdx] = grayValue;
    image_ptr[baseIdx + 1] = grayValue; image_ptr[baseIdx + 2] = grayValue; image_ptr[baseIdx + 3] = 0;
  }
};

int main() {
  int const BODY_COUNT = 10e6;
  int const IMAGE_DIM = 1000;
  int const BODY_COUNT_PIXEL_SIZE = IMAGE_DIM * IMAGE_DIM;
  int const RGBA_IMAGE_SIZE = IMAGE_DIM * PIXEL_RGBA_RATIO; // image has 4 values per pixels


  // initilize bodies
  auto bodies = thrust::device_vector<Body>(BODY_COUNT);
  auto index_sequence_begin = thrust::counting_iterator<unsigned int>(0);

  thrust::transform(
      index_sequence_begin,
      index_sequence_begin + BODY_COUNT,
      bodies.begin(),
      initRandomPrg()
      );

  std::cout << "Initialized Bodies" << "\n\n";

  // initialize pixel counts 
  auto pixelCounts = thrust::device_vector<int>(BODY_COUNT_PIXEL_SIZE);
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
    index_sequence_begin + BODY_COUNT_PIXEL_SIZE,
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
    auto alpha = (int)deviceImage[baseIdx + 3]; 

    std::cout << b.x << '\t' << b.y << '\t';
    std::cout << offset << ' ' <<  bc.x << ',' << bc.y << "\t\t";
    std::cout << pixelCounts[offset] << "\t\t";
    std::cout << red << ", " <<  blue << ", " << green <<", " << alpha;
    std::cout << '\n';
  }

  auto max = thrust::reduce(
      pixelCounts.begin(), pixelCounts.end(),
      0, thrust::maximum<int>()
  );

  std::cout << "Max Pixel Count: \t" << max << '\n';
  std::cout << "Body Count: \t\t" << BODY_COUNT << '\n';

  return 0;
}
