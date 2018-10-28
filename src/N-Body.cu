#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <ctime>
#include <cmath>

const float EPS2 = M_E * M_E;

/**
 * Struct representing a body
 */
struct Body {
	float3 position, velocity;

	__host__ __device__ Body(float3 _position, float3 _velocity):
			position(_position), velocity(_velocity) {};

	__host__ __device__ Body() : position(float3{0., 0., 0.}), velocity(float3{0., 0., 0.}) {};
};

/**
 * Image Coordinates
 */
struct ImageCoord {
	int x, y, imageDim;

	__host__ __device__
	// create coordinate from components
	ImageCoord(int _x, int _y, int _imageDim) :
			x(_x), y(_y), imageDim(_imageDim) {
	};

	__host__ __device__
	// default initialization
	ImageCoord() :
			x(0), y(0), imageDim(0) {
	};

	__host__ __device__
	// create image coordinate from a body's position
	ImageCoord(const Body &body, int _imageDim) :
			imageDim(_imageDim) {
		auto halfImageDim = imageDim / 2;
		x = (body.position.x + 1.) * halfImageDim;
		y = (body.position.y + 1.) * halfImageDim;
	};

	__host__ __device__
	int toOffset() {
		return x + y * imageDim;
	};
};

/**
 * Program to initialize bodies with random values between -1,1
 */
struct initRandomPrg {
	float minValue, maxValue;
	int seed;

	__host__ __device__ initRandomPrg(int _seed = 0, float _mnV = -1.f,
			float _mxV = 1.f) :
			seed(_seed), minValue(_mnV), maxValue(_mxV) {
	};

	__host__  __device__ Body operator()(const unsigned int idx) const {
		thrust::default_random_engine rng(seed);
		thrust::uniform_real_distribution<float> dist(minValue, maxValue);
		rng.discard(idx);

		return Body {float3{dist(rng), dist(rng), dist(rng)},
					float3{dist(rng), dist(rng),dist(rng)}};
	}
};

struct calculateAcceleration {
	const Body *bodies;
	const int bodyCount;

	__host__ __device__ calculateAcceleration(
			const Body *_bodies, int _bodyCount):
			bodies(_bodies), bodyCount(_bodyCount) {};

	__host__  __device__ float3 operator()(const int idx) {
		float3 result { 0., 0., 0. };
		auto body = bodies[idx];

		for (int i = 0; i < bodyCount; i++) {
			auto otherBody = bodies[i];
			if (idx == i) {
				continue;
			}


			auto position = otherBody.position;
			auto diffX = body.position.x - position.x;
			auto diffY = body.position.y - position.y;
			auto diffZ = body.position.z - position.z;
			auto distSqr = diffX * diffX +
					diffY * diffY +
					diffZ * diffZ + EPS2;
			auto distSixth = distSqr * distSqr * distSqr;
			auto invDistCube = 1.0f / sqrtf(distSixth);

			result.x += invDistCube * diffX;
			result.y += invDistCube * diffY;
			result.z += invDistCube * diffZ;
		}

		return result;
	}
};

int main(int argc, char **argv) {
	int const BODY_COUNT = 10e4;

	// initialize bodies
	auto bodies = thrust::device_vector<Body>(BODY_COUNT);
	auto bodies_ptr = thrust::raw_pointer_cast(bodies.data());
	auto index_sequence_begin = thrust::counting_iterator<unsigned int>(0);

	thrust::transform(index_sequence_begin, index_sequence_begin + BODY_COUNT,
			bodies.begin(), initRandomPrg(std::time(0)));

	std::cout << "Initialized Bodies" << "\n\n";

	// calculate forces
	index_sequence_begin = thrust::counting_iterator<unsigned int>(0);
	auto forces = thrust::device_vector<float3>(BODY_COUNT);
	auto h_forces = thrust::host_vector<float3>(BODY_COUNT);
	thrust::transform(
      index_sequence_begin, index_sequence_begin + BODY_COUNT,
			forces.begin(), calculateAcceleration(bodies_ptr, BODY_COUNT));

	std::cout << "Calculate Initial Forces" << "\n\n";

//	thrust::copy(forces.begin(), forces.end(), h_forces.begin());
	
//	std::cout << "Copied" << "\n\n";
//
//	for (int i = 0; i < 20; i++) {
//		auto force = h_forces[i];
//		std::cout << force.x << " " << force.y << " " << force.z << '\n';
//	}
//
//	std::cout << "Return" << "\n\n";

	return 0;
}
