/*
 * A Much Simple Network Architecture (without processing direction encoding)
 * (mainly from: https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/network_with_input_encoding.h)
 */

#pragma once

#include <tiny-cuda-nn/network_with_input_encoding.h>

NGP_NAMESPACE_BEGIN

namespace MatrixFunc {
	enum class MatrixOperation {
		CopyOverwrite,  // copy and overwrite to the dst
		AssignConstant  // assign constant value to the dst
	};

	template <MatrixOperation matrix_operation, typename T>
	__global__ void submatrix_kernel(
		const uint32_t n_elements,
		T* __restrict__ mat, const uint32_t stride, bool cm,
		const uint32_t n_rows, const uint32_t i_start_dst, const uint32_t i_start_src,
		const T value
	) {
		const uint32_t ij = threadIdx.x + blockIdx.x * blockDim.x;
		if (ij >= n_elements) return;

		// column major friendly
		const uint32_t j_offset = ij / n_rows;
		const uint32_t i_offset = ij - j_offset * n_rows;

		auto indexing = [&](uint32_t i, uint32_t j) {
			if (cm) return i + j * stride;
			return i * stride + j;
		};

		const uint32_t ind_dst = indexing(i_offset + i_start_dst, j_offset);
		const uint32_t ind_src = indexing(i_offset + i_start_src, j_offset);

		if (matrix_operation == MatrixOperation::CopyOverwrite) {
			mat[ind_dst] = T(mat[ind_src]); // selfcopy
		} else {
			mat[ind_dst] = value;
		}
	}

	template <typename T>
	void zeros_(cudaStream_t stream, const tcnn::GPUMatrixDynamic<T>& mat, const uint32_t i_start, const uint32_t n_rows) {
		tcnn::linear_kernel(submatrix_kernel<MatrixOperation::AssignConstant, T>, 0, stream,
			n_rows * mat.n(),
			mat.data(), mat.stride(), mat.layout() == tcnn::MatrixLayout::ColumnMajor,
			n_rows, i_start, 0, T(0)
		);
	}

	template <typename T>
	void selfcopy_(cudaStream_t stream, const tcnn::GPUMatrixDynamic<T>& mat, const uint32_t i_start_dst, const uint32_t i_start_src, const uint32_t n_rows) {
		tcnn::linear_kernel(submatrix_kernel<MatrixOperation::CopyOverwrite, T>, 0, stream,
			n_rows * mat.n(),
			mat.data(), mat.stride(), mat.layout() == tcnn::MatrixLayout::ColumnMajor,
			n_rows, i_start_dst, i_start_src, T(0)
		);
	}
}

//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=

template <typename T>
class NerfNetwork : public tcnn::NetworkWithInputEncoding<T> {
public:
	using json = nlohmann::json;
	using Parent = tcnn::NetworkWithInputEncoding<T>;

	NerfNetwork(const json& pos_encoding, const json& density_network): Parent(3, 4, pos_encoding, density_network) { }
	virtual ~NerfNetwork() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		auto input_pos = input.slice_rows(0, 3);
		Parent::inference_mixed_precision_impl(stream, input_pos, output, use_inference_params);
	}

	std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		auto input_pos = input.slice_rows(0, 3);
		return Parent::forward_impl(stream, input_pos, output, use_inference_params, prepare_input_gradients);
	}

	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
		auto input_pos = input.slice_rows(0, 3);
		tcnn::GPUMatrixDynamic<float> dL_dinput_pos{nullptr, 0, 0};
		if (dL_dinput) {
			dL_dinput_pos = dL_dinput->slice_rows(0, 3);
		}
		// zero out other grad
		MatrixFunc::zeros_(stream, dL_doutput, 4, this->padded_output_width() - 4);
		// compute
		Parent::backward_impl(stream, ctx, input_pos, output, dL_doutput, dL_dinput ? &dL_dinput_pos : nullptr, use_inference_params, param_gradients_mode);
		// zero out dir grad
		if (dL_dinput) {
			MatrixFunc::zeros_(stream, *dL_dinput, 4, 3);
		}
	}

	void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::MatrixLayout::ColumnMajor) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}
		this->inference_mixed_precision_impl(stream, input, output, use_inference_params); // don't check the width
		// copy the forth density to the first position
		MatrixFunc::selfcopy_(stream, output, 0, 3, 1);
	}

	uint32_t padded_density_output_width() const {
		return this->padded_output_width();
	}

	uint32_t input_width() const override {
		return 7;
	}

	uint32_t output_width() const override {
		return 4;
	}

	uint32_t n_extra_dims() const {
		return 0;
	}
};

NGP_NAMESPACE_END
