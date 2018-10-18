#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>

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

struct initRandomPrg
{
  float minValue, maxValue;

  __host__ __device__
  initRandomPrg(float _mnV=0.f, float _mxV=1.f) : minValue(_mnV), maxValue(_mxV) {};

  __host__ __device__
  Body operator()(const unsigned int n) const
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

struct addPrg
{
  __host__ __device__
  addPrg() {};

  __host__ __device__
  Body operator()(const Body& a, const Body& b) const
  {
    Body result(a);
    result.x += b.x;
    result.y += b.y;
    return result;
  }
};


int main() {
  int N = 8000 * 8000; // 800px x 800px image
  int iterations = 10;

  auto x = thrust::device_vector<Body>(N);
  auto y = thrust::device_vector<Body>(N);
  auto output = thrust::device_vector<Body>(N);

  // initilize array  
  auto index_sequence_begin = thrust::counting_iterator<unsigned int>(0);
  
  thrust::transform(
      index_sequence_begin,
      index_sequence_begin + N,
      x.begin(),
      initRandomPrg()
  );
  
  thrust::transform(
      index_sequence_begin,
      index_sequence_begin + N,
      y.begin(),
      initRandomPrg()
  );


  // add them up
  for (int i = 0; i < iterations; i++) {
    thrust::transform(
        x.begin(), x.end(),
        y.begin(),
        output.begin(),
        //thrust::plus<float>()
        addPrg()
    );
  }

  for (int i = 0; i < 10; i++) {
    Body b = output[i];
    std::cout << b.x << '\n';
  }

  return 0;
}
