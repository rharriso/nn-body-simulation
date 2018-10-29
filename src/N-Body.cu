#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>

#include <fmt/format.h>
#include <ctime>
#include <cmath>

const float SOFTENING = 1e-9f;
const int BLOCK_SIZE = 256;


/**
 * Struct representing a body
 */
struct Body {
	float3 position, velocity;
	float mass;

	__host__ __device__ Body(float3 _position, float3 _velocity, float _mass):
			position(_position), velocity(_velocity), mass(_mass) {};

	__host__ __device__ Body() :
			position(float3{0., 0., 0.}),
			velocity(float3{0., 0., 0.}),
			mass{1.} {};
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

		return Body {
			float3{dist(rng), dist(rng), dist(rng)},
			float3{dist(rng), dist(rng),dist(rng)},
			dist(rng)
		};
	}
};

__device__
float3 bodyInteraction(Body &bi, Body &bj, float3 accel) {
	float3 r;
	r.x = bi.position.x - bj.position.x;
	r.y = bi.position.y - bj.position.y;
	r.z = bi.position.z - bj.position.z;

	auto distSqr = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
	auto distSixth = distSqr * distSqr * distSqr;
	auto invDistCube = 1.0f / sqrtf(distSixth);

	auto s = bj.mass * invDistCube;
	accel.x += r.x * s;
	accel.y += r.y * s;
	accel.z += r.z * s;
	return accel;
}

__global__
void updateVelocities(Body *bodies, int bodyCount, float dt) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (bodyCount <= idx) return;

	float3 accel{0., 0., 0.};
	auto body = bodies[idx];

	for(int tile = 0; tile < gridDim.x; tile++) {
		__shared__ Body sharedBodies[BLOCK_SIZE];
		Body tBody = bodies[tile * blockDim.x + threadIdx.x];
		sharedBodies[threadIdx.x] = Body{tBody.position, tBody.velocity, tBody.mass};
		__syncthreads();

		for(int j = 0; j < BLOCK_SIZE; j++) {
			accel = bodyInteraction(body, sharedBodies[j], accel);
		}

		__syncthreads();
	}

	// update blockBody velocity
	body.velocity.x += accel.x * dt;
	body.velocity.y += accel.y * dt;
	body.velocity.z += accel.z * dt;
}


int main(int argc, char **argv) {
	int const BODY_COUNT = 1000;
	const float dt = 0.01f;
	int numBlocks = (BODY_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// initialize bodies
	auto bodies = thrust::device_vector<Body>(BODY_COUNT);
	auto bodies_ptr = thrust::raw_pointer_cast(bodies.data());
	auto index_sequence_begin = thrust::counting_iterator<unsigned int>(0);

	thrust::transform(index_sequence_begin, index_sequence_begin + BODY_COUNT,
			bodies.begin(), initRandomPrg(std::time(0)));

	std::cout << fmt::format("Initialize Bodies: {}\n\n", BODY_COUNT);

	// calculate forces
	index_sequence_begin = thrust::counting_iterator<unsigned int>(0);
	auto forces = thrust::device_vector<float3>(BODY_COUNT);
	auto forces_ptr = thrust::raw_pointer_cast(forces.data());
	auto h_forces = thrust::host_vector<float3>(BODY_COUNT);

	std::cout << "Calculate Velocities" << "\n\n";

	updateVelocities<<<numBlocks, BLOCK_SIZE>>>(bodies_ptr, BODY_COUNT, dt);

	cudaDeviceSynchronize();

	std::cout << "Return" << "\n\n";

	return 0;
}
