/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_buffer.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/gpu_memory.h>

#include <memory>
#include <vector>

NGP_NAMESPACE_BEGIN

typedef unsigned int GLenum;
typedef int          GLint;
typedef unsigned int GLuint;

class SurfaceProvider {
public:
	virtual cudaSurfaceObject_t surface() = 0;
	virtual cudaArray_t array() = 0;
	virtual Eigen::Vector2i resolution() const = 0;
	virtual void resize(const Eigen::Vector2i&) = 0;
};

class CudaSurface2D : public SurfaceProvider {
public:
	CudaSurface2D() {
		m_array = nullptr;
		m_surface = 0;
	}

	~CudaSurface2D() {
		free();
	}

	void free();

	void resize(const Eigen::Vector2i& size) override;

	cudaSurfaceObject_t surface() override {
		return m_surface;
	}

	cudaArray_t array() override {
		return m_array;
	}

	Eigen::Vector2i resolution() const override {
		return m_size;
	}

private:
	Eigen::Vector2i m_size = Eigen::Vector2i::Constant(0);
	cudaArray_t m_array;
	cudaSurfaceObject_t m_surface;
};

class CudaRenderBuffer {
public:
	CudaRenderBuffer(const std::shared_ptr<SurfaceProvider>& surf) : m_surface_provider{surf} {}

	CudaRenderBuffer(const CudaRenderBuffer& other) = delete;
	CudaRenderBuffer(CudaRenderBuffer&& other) = default;

	cudaSurfaceObject_t surface() {
		return m_surface_provider->surface();
	}

	Eigen::Vector2i in_resolution() const {
		return m_in_resolution;
	}

	Eigen::Vector2i out_resolution() const {
		return m_surface_provider->resolution();
	}

	void resize(const Eigen::Vector2i& res);

	void reset_accumulation() {
		m_spp = 0;
	}

	uint32_t spp() const {
		return m_spp;
	}

	Eigen::Array4f* frame_buffer() const {
		return m_frame_buffer.data();
	}

	float* depth_buffer() const {
		return m_depth_buffer.data();
	}

	Eigen::Array4f* accumulate_buffer() const {
		return m_accumulate_buffer.data();
	}

	void clear_frame(cudaStream_t stream);

	void accumulate(float exposure, cudaStream_t stream);

	void tonemap(float exposure, const Eigen::Array4f& background_color, EColorSpace output_color_space, cudaStream_t stream);

	void overlay_image(
		float alpha,
		const Eigen::Array3f& exposure,
		const Eigen::Array4f& background_color,
		EColorSpace output_color_space,
		const void* __restrict__ image,
		EImageDataType image_data_type,
		const Eigen::Vector2i& resolution,
		int fov_axis,
		float zoom,
		const Eigen::Vector2f& screen_center,
		cudaStream_t stream
	);

	void overlay_false_color(Eigen::Vector2i training_resolution, bool to_srgb, int fov_axis, cudaStream_t stream, const float *error_map, Eigen::Vector2i error_map_resolution, const float *average, float brightness, bool viridis);

	SurfaceProvider& surface_provider() {
		return *m_surface_provider;
	}

	void set_color_space(EColorSpace color_space) {
		if (color_space != m_color_space) {
			m_color_space = color_space;
			reset_accumulation();
		}
	}

	void set_tonemap_curve(ETonemapCurve tonemap_curve) {
		if (tonemap_curve != m_tonemap_curve) {
			m_tonemap_curve = tonemap_curve;
			reset_accumulation();
		}
	}

private:
	uint32_t m_spp = 0;
	EColorSpace m_color_space = EColorSpace::Linear;
	ETonemapCurve m_tonemap_curve = ETonemapCurve::Identity;

	Eigen::Vector2i m_in_resolution = Eigen::Vector2i::Zero();

	tcnn::GPUMemory<Eigen::Array4f> m_frame_buffer;
	tcnn::GPUMemory<float> m_depth_buffer;
	tcnn::GPUMemory<Eigen::Array4f> m_accumulate_buffer;

	std::shared_ptr<SurfaceProvider> m_surface_provider;
};

NGP_NAMESPACE_END
