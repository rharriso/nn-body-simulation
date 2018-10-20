#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>

/**
 * Struct representing a body 
 */
struct Body {
  float x, y, vx, vy;

  __host__ __device__
    Body (const Body &otherBody): 
      x(otherBody.x),
      y(otherBody.y),
      vx(otherBody.vx),
      vy(otherBody.vy) {};
  __host__ __device__
    Body (float _x, float _y, float _vx, float _vy): x(_x), y(_y), vx(_vx), vy(_vy) {};
  __host__ __device__
    Body () {};
};

/**
 * Image Coordinates
 */
struct ImageCoord {
  unsigned int x, y, imageDim;

  __host__ __device__
    ImageCoord (
        unsigned int _x,
        unsigned int _y,
        unsigned int _imageDim
        ): x(_x), y(_y), imageDim(_imageDim) {};

  __host__ __device__
    ImageCoord (): x(0), y(0), imageDim(0) {};

  __host__ __device__
    ImageCoord (
        const Body &body,
        unsigned int _imageDim
        ):  imageDim(_imageDim) {
      auto halfImageDim = imageDim / 2;
      x = (body.x + 1.) * halfImageDim;
      y = (body.y + 1.) * halfImageDim;
    };

  __host__
    ImageCoord (
        const Body &body,
        unsigned int _imageDim,
        bool ok
        ):  imageDim(_imageDim) {
      auto halfImageDim = imageDim / 2;
      //std::cout << "IMAGE_DIM: " << imageDim << '\n';
      //std::cout << "HALFDIM: " << halfImageDim << '\n';
      x = (body.x + 1.) * halfImageDim;
      y = (body.y + 1.) * halfImageDim;
      //std::cout << "X: " << x << " Y: " << y << '\n';
    };


  __host__ __device__
    unsigned int toOffset() {
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
    initRandomPrg(float _mnV=-1.f, float _mxV=1.f):
      minValue(_mnV), maxValue(_mxV) {};

  __host__ __device__
    Body operator()(const int n) const
    {
      thrust::default_random_engine rng;
      thrust::uniform_real_distribution<float> dist(minValue, maxValue);
      rng.discard(n);

      return Body{
        dist(rng), dist(rng),
          dist(rng), dist(rng)
      };
    }
};


struct mapBodyToPixelCounts
{
  unsigned int *pixelCounts;
  const unsigned int imageDim;

  __host__ __device__
    mapBodyToPixelCounts(
        unsigned int _imageDim,
        unsigned int *_pixelCounts
        ): imageDim(_imageDim), pixelCounts(_pixelCounts) {};

  __device__
    void operator()(const Body &body) const
    {
      auto offset = ImageCoord(body, imageDim).toOffset();
      atomicAdd(&pixelCounts[offset], 1);
    }
};


int main() {
  unsigned int const BODY_COUNT = 10e6;
  unsigned int const IMAGE_DIM = 2000;
  unsigned int const IMG_VECTOR_SIZE = IMAGE_DIM * IMAGE_DIM;

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
  auto pixelCounts = thrust::device_vector<unsigned int>(IMG_VECTOR_SIZE);
  thrust::fill(pixelCounts.begin(), pixelCounts.end(), 0);

  std::cout << "Initialized Pxels" << "\n\n";

  // get pixel counts
  thrust::for_each(
      bodies.begin(),
      bodies.end(),
      mapBodyToPixelCounts(IMAGE_DIM, thrust::raw_pointer_cast(pixelCounts.data()))
      );

  std::cout << "Mapped Points to Pixels" << "\n\n";

  for (auto i = 0; i < 10; i++) {
    //ImageCoord bc = bodyCoords[i];
    //std::cout << bc.x << ' ' << bc.y << "  ";
    Body b = bodies[i];
    auto offset = ImageCoord(b, IMAGE_DIM, true).toOffset();

    std::cout << b.x << '\t' << b.y << '\t';
    std::cout << offset << ' ' << IMG_VECTOR_SIZE << '\t';
    std::cout << pixelCounts[offset];
    std::cout << '\n';
  }

  auto max = thrust::reduce(
      pixelCounts.begin(), pixelCounts.end(),
      0, thrust::maximum<unsigned int>()
  );

  std::cout << "Max Pixel Count: " << max << '\n';

  return 0;
}
