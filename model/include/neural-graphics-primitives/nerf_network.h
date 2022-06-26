/* This Network Implementation Has:
 *	- 3 MLPs:
 *	  - density network: to encode positional embedding, as well as density prediction
 *	  - rgb network: to get rgb prediction
 *	  - attr network: to get attribute prediction
 *	
 *	- 3 encoding modules
 *	  - density encoding: to encode positional embedding via hash encoder
 *	  - rgb encoding: to encode direction, it can be disabled by passing Identity module with scale=0
 *	  - attr encoding: to encode direction, it can be disabled by passing Identity module with scale=0
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>

NGP_NAMESPACE_BEGIN

//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=

enum MatrixOperation { 
	CopyOverwrite,  // copy the src and overwrite to the dst
	CopyAccumulate, // copy the src and accumulate to the dst
	AssignConstant  // assign constant value to the dst
};

// universal matrix operation kernel
template <MatrixOperation matrix_operation=CopyOverwrite, typename T_dst, typename T_src>
__global__ void submatrix_op(const uint32_t n_elements,
							 // dst
							 T_dst* __restrict__ dst, const uint32_t stride_dst, bool cm_dst,
							 const uint32_t i_start_dst, const uint32_t j_start_dst,
							 const uint32_t i_end_dst, const uint32_t j_end_dst,
							 // src
							 T_src* __restrict__ src, const uint32_t stride_src, bool cm_src,
							 const uint32_t i_start_src, const uint32_t j_start_src,
							 // constant
							 const T_dst value
) {
	const uint32_t ij = threadIdx.x + blockIdx.x * blockDim.x;
	if (ij >= n_elements) return;

	const uint32_t n_rows = i_end_dst - i_start_dst;
	// column major friendly
	const uint32_t j_offset = ij / n_rows;
	const uint32_t i_offset = ij - j_offset * n_rows;

	const uint32_t i_dst = i_offset + i_start_dst;
	const uint32_t j_dst = j_offset + j_start_dst;

	const uint32_t i_src = i_offset + i_start_src;
	const uint32_t j_src = j_offset + j_start_src;

	auto indexing = [](uint32_t i, uint32_t j, uint32_t stride, bool cm) {
		if (cm) return i + j * stride;
		return i * stride + j;
	};

	const uint32_t ind_src = indexing(i_src, j_src, stride_src, cm_src);
	const uint32_t ind_dst = indexing(i_dst, j_dst, stride_dst, cm_dst);

	switch (matrix_operation) {
		case CopyOverwrite: // copy one element
			dst[ind_dst] = T_dst(src[ind_src]);
			break;
		case CopyAccumulate: // accumulate one element
			dst[ind_dst] += T_dst(src[ind_src]);
			break;
		case AssignConstant: // assign a constant
			dst[ind_dst] = value;
			break;
	}
}

// a row-wise matrix op launcher
template <MatrixOperation matrix_operation, typename T_dst, typename T_src=T_dst>
void submatrix_row_op(cudaStream_t stream,
					  const uint32_t n_rows, const uint32_t batch_size, 
				      const tcnn::GPUMatrixDynamic<T_dst>& mat_dst, const uint32_t i_start_dst,
				      const tcnn::GPUMatrixDynamic<T_dst>& mat_src = {nullptr, 0, 0}, const uint32_t i_start_src = 0u,
				      const T_dst value = T_dst(0)
) {
	tcnn::linear_kernel(submatrix_op<matrix_operation, T_dst, T_src>, 0, stream,
		n_rows * batch_size,
		mat_dst.data(), mat_dst.stride(), mat_dst.layout() == tcnn::MatrixLayout::ColumnMajor,
		i_start_dst, 0u, i_start_dst + n_rows, batch_size,
		mat_src.data(), mat_src.stride(), mat_src.layout() == tcnn::MatrixLayout::ColumnMajor,
		i_start_src, 0u,
		value
	);
}

//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=

template <typename T>
class NerfNetwork : public tcnn::Network<float, T> {
public:
	using json = nlohmann::json;
	using MatrixT = tcnn::GPUMatrixDynamic<T>;
	using MatrixF = tcnn::GPUMatrixDynamic<float>;

	NerfNetwork(const json& rgb_encoding, const json& density_encoding, const json& attr_encoding,
				const json& rgb_network,  const json& density_network,  const json& attr_network 
	) {
		
		// dims of attribute
		m_n_attr_dims = attr_network.value("n_attr", 0u);
		
		// calculate alignment
		auto density_alignment = density_network.contains("otype") && (
			tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || 
			tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")
		) ? 16u : 8u ;
		auto rgb_alignment = tcnn::minimum_alignment(rgb_network);
		auto attr_alignment = (m_n_attr_dims > 0) ? tcnn::minimum_alignment(attr_network) : rgb_alignment;
		if (density_alignment != rgb_alignment || rgb_alignment != attr_alignment) {
			// same alignments will make it easy to deal with
			throw std::runtime_error("Expected all alignments must be the same");
		}
		const uint32_t alignment = rgb_alignment;
		
		// create encoding modules
		m_density_encoding.reset(tcnn::create_encoding<T>(3u, density_encoding, alignment));
		m_rgb_encoding.reset(tcnn::create_encoding<T>(3u, rgb_encoding, alignment));
		if (m_n_attr_dims > 0) {
			m_attr_encoding.reset(tcnn::create_encoding<T>(3u, attr_encoding, alignment));
		}

		// create density network
		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_density_encoding->padded_output_width();
		local_density_network_config["n_output_dims"] = density_network.value("n_output_dims", 16);
		m_density_network.reset(tcnn::create_network<T>(local_density_network_config));

		// create rgb network
		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_density_network->padded_output_width() + m_rgb_encoding->padded_output_width();
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));

		// create attr network
		if (m_n_attr_dims > 0) {
			json local_attr_network_config = attr_network;
			local_attr_network_config["n_input_dims"] = m_density_network->padded_output_width() + m_attr_encoding->padded_output_width();
			local_attr_network_config["n_output_dims"] = m_n_attr_dims;
			m_attr_network.reset(tcnn::create_network<T>(local_attr_network_config));
		}
	}

	virtual ~NerfNetwork() { }

	void inference_mixed_precision_impl(
		cudaStream_t stream, 
		const MatrixF& input, 
		MatrixT& output, 
		bool use_inference_params = true
	) override {
		auto batch_size = input.n();
		
		// allocate workspace
		auto temp = TempBuffer(this, batch_size, stream, output.layout() == tcnn::MatrixLayout::ColumnMajor);

		// input
		auto input_pos = input.slice_rows(0, 3u);
		auto input_dir = input.slice_rows(4u, 3u);

		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// DENSITY
		m_density_encoding->inference_mixed_precision(stream,
			input_pos,
			temp.density_network_input,
			use_inference_params
		);
		m_density_network->inference_mixed_precision(stream, 
			temp.density_network_input,
			temp.density_network_output,
			use_inference_params
		);

		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// RGB

		// concatenate with density network output
		submatrix_row_op<CopyOverwrite, T>(stream,
			m_density_network->padded_output_width(), batch_size,
			temp.rgb_network_input, 0u, // to `rgb_network_input`
			temp.density_network_output
		);

		// forward
		auto rgb_encoding_output = temp.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_rgb_encoding->padded_output_width());
		m_rgb_encoding->inference_mixed_precision(stream,
			input_dir, // dir
			rgb_encoding_output,
			use_inference_params
		);
		m_rgb_network->inference_mixed_precision(stream, 
			temp.rgb_network_input,
			temp.rgb_network_output,
			use_inference_params
		);
		
		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// ATTR

		if (m_n_attr_dims > 0) {
			// concatenate with density network output
			submatrix_row_op<CopyOverwrite, T>(stream,
				m_density_network->padded_output_width(), batch_size,
				temp.attr_network_input, 0u, // to `attr_network_input`
				temp.density_network_output
			);

			// forward
			auto attr_encoding_output = temp.attr_network_input.slice_rows(m_density_network->padded_output_width(), m_attr_encoding->padded_output_width());
			m_attr_encoding->inference_mixed_precision(stream,
				input_dir, // dir
				attr_encoding_output, 
				use_inference_params
			);
			m_attr_network->inference_mixed_precision(stream, 
				temp.attr_network_input,
				temp.attr_network_output,
				use_inference_params
			);
		}
		
		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// gather output

		submatrix_row_op<CopyOverwrite, T>(stream,
			3u, batch_size,
			output, 0u,
			temp.rgb_network_output // rgb
		);
		submatrix_row_op<CopyOverwrite, T>(stream,
			1u, batch_size,
			output, 3u,
			temp.density_network_output // density
		);
		submatrix_row_op<CopyOverwrite, T>(stream,
			m_n_attr_dims, batch_size,
			output, 4u,
			temp.attr_network_output // attr
		);

	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream, 
		const MatrixF& input, 
		MatrixT* output = nullptr, 
		bool use_inference_params = false, 
		bool prepare_input_gradients = false
	) override {
		auto batch_size = input.n();

		// allocate workspace
		auto ctx = std::make_unique<ForwardContext>(this, batch_size, stream, 
			output ? output->layout() == tcnn::MatrixLayout::ColumnMajor : true
		);

		// input
		auto input_pos = input.slice_rows(0, 3u);
		auto input_dir = input.slice_rows(4u, 3u);

		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// DENSITY

		ctx->density_encoding_ctx = m_density_encoding->forward(stream,
			input_pos,
			&ctx->density_network_input,
			use_inference_params, prepare_input_gradients
		);
		
		ctx->density_network_ctx = m_density_network->forward(stream, 
			ctx->density_network_input, 
			&ctx->density_network_output,
			use_inference_params, prepare_input_gradients
		);

		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// RGB

		// concatenate with density network output
		submatrix_row_op<CopyOverwrite, T>(stream,
			m_density_network->padded_output_width(), batch_size,
			ctx->rgb_network_input, 0u, // to `rgb_network_input`
			ctx->density_network_output
		);

		// forward
		auto rgb_encoding_output = ctx->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_rgb_encoding->padded_output_width());
		ctx->rgb_encoding_ctx = m_rgb_encoding->forward(stream,
			input_dir,
			&rgb_encoding_output,
			use_inference_params, prepare_input_gradients
		);

		ctx->rgb_network_ctx = m_rgb_network->forward(stream, 
			ctx->rgb_network_input, 
			output ? &ctx->rgb_network_output : nullptr, 
			use_inference_params, prepare_input_gradients
		);

		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// ATTR

		if (m_n_attr_dims > 0) {

			// concatenate with density network output
			submatrix_row_op<CopyOverwrite, T>(stream,
				m_density_network->padded_output_width(), batch_size,
				ctx->attr_network_input, 0u, // to `attr_network_input`
				ctx->density_network_output
			);

			// forward
			auto attr_encoding_output = ctx->attr_network_input.slice_rows(m_density_network->padded_output_width(), m_attr_encoding->padded_output_width());
			ctx->attr_encoding_ctx = m_attr_encoding->forward(stream,
				input_dir,
				&attr_encoding_output,
				use_inference_params, prepare_input_gradients
			);
			ctx->attr_network_ctx = m_attr_network->forward(stream,
				ctx->attr_network_input, 
				output ? &ctx->attr_network_output : nullptr, 
				use_inference_params, prepare_input_gradients
			);
		}

		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// gather output

		if (output) {
			submatrix_row_op<CopyOverwrite, T>(stream,
				3u, batch_size,
				*output, 0u,
				ctx->rgb_network_output // rgb
			);
			submatrix_row_op<CopyOverwrite, T>(stream,
				1u, batch_size,
				*output, 3u,
				ctx->density_network_output // density
			);
			submatrix_row_op<CopyOverwrite, T>(stream,
				m_n_attr_dims, batch_size,
				*output, 4u,
				ctx->attr_network_output // attr
			);
		}

		return ctx;
	}

	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const MatrixF& input,
		const MatrixT& output,
		const MatrixT& dL_doutput,
		MatrixF* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
		auto batch_size = input.n();

		// convert to our custom context
		const auto& ctx_forward = dynamic_cast<const ForwardContext&>(ctx);

		// allocate gradient buffer
		auto temp_grad = TempBuffer(this, batch_size, stream, output.layout() == tcnn::MatrixLayout::ColumnMajor);
		temp_grad.rgb_attr_network_output_zero_(stream);

		// input
		auto input_pos = input.slice_rows(0, 3u);
		auto input_dir = input.slice_rows(4u, 3u);

		// input grad
		MatrixF grad_pos{nullptr, 0, 0}, grad_dir{nullptr, 0, 0};
		if (dL_dinput) {
			grad_pos = dL_dinput->slice_rows(0, 3u);
			grad_dir = dL_dinput->slice_rows(4u, 3u);
		}

		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// RGB

		// copy rgb gradient
		submatrix_row_op<CopyOverwrite, T>(stream,
			3u, batch_size,
			temp_grad.rgb_network_output, 0u,
			dL_doutput // rgb
		);

		// backward
		m_rgb_network->backward(stream, *ctx_forward.rgb_network_ctx, 
			ctx_forward.rgb_network_input,  // input
			ctx_forward.rgb_network_output, // output
			temp_grad.rgb_network_output, // dL_doutput
			&temp_grad.rgb_network_input, // dL_dinput
			use_inference_params, param_gradients_mode
		);
		if (m_rgb_encoding->n_params() > 0 || dL_dinput) {
			m_rgb_encoding->backward(stream, *ctx_forward.rgb_encoding_ctx,
				input_dir, // input
				ctx_forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_rgb_encoding->padded_output_width()), // output
				temp_grad.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_rgb_encoding->padded_output_width()),   // dL_doutput
				dL_dinput ? &grad_dir : nullptr, // dL_dinput (NOTE: directly overwrite to `grad_dir`)
				use_inference_params, param_gradients_mode
			);
		}

		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// ATTR

		if (m_n_attr_dims > 0) {
			// copy attr gradient
			submatrix_row_op<CopyOverwrite, T>(stream,
				m_n_attr_dims, batch_size,
				temp_grad.attr_network_output, 0u,
				dL_doutput, 4u // attr
			);

			// backward
			m_attr_network->backward(stream, *ctx_forward.attr_network_ctx, 
				ctx_forward.attr_network_input,  // input
				ctx_forward.attr_network_output, // output
				temp_grad.attr_network_output, // dL_doutput
				&temp_grad.attr_network_input, // dL_dinput
				use_inference_params, param_gradients_mode
			);
			if (m_attr_encoding->n_params() > 0 || dL_dinput) {
				auto grad_dir_from_attr_encoding = MatrixF{3, batch_size, stream, dL_dinput->layout()};
				m_attr_encoding->backward(stream, *ctx_forward.attr_encoding_ctx,
					input_dir, // input
					ctx_forward.attr_network_input.slice_rows(m_density_network->padded_output_width(), m_attr_encoding->padded_output_width()), // out
					temp_grad.attr_network_input.slice_rows(m_density_network->padded_output_width(), m_attr_encoding->padded_output_width()),   // dL_doutput
					dL_dinput ? &grad_dir_from_attr_encoding : nullptr, // dL_dinput (NOTE: write to `grad_dir_from_attr_encoding` first, then we accumulate it to `grad_dir`)
					use_inference_params, param_gradients_mode
				);
				// accumulate direction gradient
				submatrix_row_op<CopyAccumulate, float>(stream,
					3u, batch_size,
					grad_dir, 0u,
					grad_dir_from_attr_encoding
				);
			}
		}

		//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-
		// DENSITY

		// accumulate density gradient
		submatrix_row_op<CopyOverwrite, T>(stream,
			m_density_network->padded_output_width(), batch_size,
			temp_grad.density_network_output, 0u,
			temp_grad.rgb_network_input // rgb
		);
		submatrix_row_op<CopyAccumulate, T>(stream,
			1u, batch_size,
			temp_grad.density_network_output, 0u,
			dL_doutput, 3u // density
		);
		if (m_n_attr_dims > 0) {
			submatrix_row_op<CopyAccumulate, T>(stream,
				m_density_network->padded_output_width(), batch_size,
				temp_grad.density_network_output, 0u,
				temp_grad.attr_network_input // attr
			);
		}

		// backward
		m_density_network->backward(stream, *ctx_forward.density_network_ctx, 
			ctx_forward.density_network_input,  // input
			ctx_forward.density_network_output, // output
			temp_grad.density_network_output,   // dL_doutput
			(m_density_encoding->n_params() > 0 || dL_dinput) ? &temp_grad.density_network_input : nullptr, // dL_dinput
			use_inference_params, param_gradients_mode
		);
		if (m_density_encoding->n_params() > 0 || dL_dinput) {
			m_density_encoding->backward(stream, *ctx_forward.density_encoding_ctx,
				input_pos,
				ctx_forward.density_network_input,
				temp_grad.density_network_input,
				dL_dinput ? &grad_pos : nullptr,
				use_inference_params, param_gradients_mode
			);
		}
	}

	void density(
		cudaStream_t stream, 
		const MatrixF& input, 
		MatrixT& output, 
		bool use_inference_params = true
	) {
		if (input.layout() != tcnn::MatrixLayout::ColumnMajor) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		auto batch_size = output.n();
		auto density_network_input = MatrixT{m_density_encoding->padded_output_width(), batch_size, stream, m_density_encoding->preferred_output_layout()};

		m_density_encoding->inference_mixed_precision(stream,
			input.slice_rows(0, 3), // pos
			density_network_input,
			use_inference_params
		);
		m_density_network->inference_mixed_precision(stream, 
			density_network_input, 
			output, 
			use_inference_params
		);
	}

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		size_t offset = 0;
		// network
		m_density_network->set_params(params + offset, inference_params + offset, backward_params + offset, gradients + offset);
		offset += m_density_network->n_params();
		m_rgb_network->set_params(params + offset, inference_params + offset, backward_params + offset, gradients + offset);
		offset += m_rgb_network->n_params();
		if (m_n_attr_dims > 0) {
			m_attr_network->set_params(params + offset, inference_params + offset, backward_params + offset, gradients + offset);
			offset += m_attr_network->n_params();
		}
		// encoding
		m_density_encoding->set_params(params + offset, inference_params + offset, backward_params + offset, gradients + offset);
		offset += m_density_encoding->n_params();
		m_rgb_encoding->set_params(params + offset, inference_params + offset, backward_params + offset, gradients + offset);
		offset += m_rgb_encoding->n_params();
		if (m_n_attr_dims > 0) {
			m_attr_encoding->set_params(params + offset, inference_params + offset, backward_params + offset, gradients + offset);
			offset += m_attr_encoding->n_params();
		}
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		// network
		m_density_network->initialize_params(rnd,
			params_full_precision + offset, params + offset, inference_params + offset, backward_params + offset, gradients + offset, scale);
		offset += m_density_network->n_params();
		m_rgb_network->initialize_params(rnd,
			params_full_precision + offset, params + offset, inference_params + offset, backward_params + offset, gradients + offset, scale);
		offset += m_rgb_network->n_params();
		if (m_n_attr_dims > 0) {
			m_attr_network->initialize_params(rnd,
				params_full_precision + offset, params + offset, inference_params + offset, backward_params + offset, gradients + offset, scale);
			offset += m_attr_network->n_params();
		}
		// encoding
		m_density_encoding->initialize_params(rnd,
			params_full_precision + offset, params + offset, inference_params + offset, backward_params + offset, gradients + offset, scale);
		offset += m_density_encoding->n_params();
		m_rgb_encoding->initialize_params(rnd,
			params_full_precision + offset, params + offset, inference_params + offset, backward_params + offset, gradients + offset, scale);
		offset += m_rgb_encoding->n_params();
		if (m_n_attr_dims > 0) {
			m_attr_encoding->initialize_params(rnd,
				params_full_precision + offset, params + offset, inference_params + offset, backward_params + offset, gradients + offset, scale);
			offset += m_attr_encoding->n_params();
		}
	}

	size_t n_params() const override {
		return m_density_encoding->n_params() + m_density_network->n_params() + \
			   m_rgb_encoding->n_params() + m_rgb_network->n_params() + \
			   ((m_n_attr_dims > 0) ? m_attr_encoding->n_params() + m_attr_network->n_params() : 0)
			;
	}

	uint32_t padded_output_width() const override {
		return 4u + m_n_attr_dims;
	}

	uint32_t input_width() const override {
		return 7u;
	}

	uint32_t output_width() const override {
		return 4u + m_n_attr_dims;
	}

	uint32_t n_extra_dims() const {
		return 0u;
	}

	uint32_t n_attr_dims() const {
		return m_n_attr_dims;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		// TODO @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ 
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		if (m_n_attr_dims > 0) {
			auto attr_layers = m_attr_network->layer_sizes();
			layers.insert(layers.end(), attr_layers.begin(), attr_layers.end());
		}
		return layers;
	}

	uint32_t width(uint32_t layer) const override {
		// TODO @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ 
		return 123;
	}

	uint32_t num_forward_activations() const override {
		// TODO @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ 
		return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
	}

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
		// TODO @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ 
		return {nullptr, tcnn::MatrixLayout::ColumnMajor};
	}

	const std::shared_ptr<tcnn::Encoding<T>>& density_encoding() const {
		return m_density_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& attr_encoding() const {
		return m_attr_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& rgb_encoding() const {
		return m_rgb_encoding;
	}

	tcnn::json hyperparams() const override {
		json density_network_hyperparams = m_density_network->hyperparams();
		density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
		return {
			{"otype", "NerfNetwork"},
			{"density_encoding", m_density_encoding->hyperparams()},
			{"rgb_encoding", m_rgb_encoding->hyperparams()},
			{"attr_encoding", m_attr_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
			{"attr_network", m_attr_network->hyperparams()},
		};
	}

private:
	// density
	std::unique_ptr<tcnn::Network<T>> m_density_network = nullptr;
	std::shared_ptr<tcnn::Encoding<T>> m_density_encoding = nullptr;
	// rgb
	std::unique_ptr<tcnn::Network<T>> m_rgb_network = nullptr;
	std::shared_ptr<tcnn::Encoding<T>> m_rgb_encoding = nullptr;
	// attr
	std::unique_ptr<tcnn::Network<T>> m_attr_network = nullptr;
	std::shared_ptr<tcnn::Encoding<T>> m_attr_encoding = nullptr;

	// dims of attr
	uint32_t m_n_attr_dims = 0u;

	// necessaries to compute results
	struct TempBuffer {
		// density
		MatrixT density_network_input{nullptr, 0, 0};
		MatrixT density_network_output{nullptr, 0, 0};
		// rgb
		MatrixT rgb_network_input{nullptr, 0, 0};
		MatrixT rgb_network_output{nullptr, 0, 0};
		// attr
		MatrixT attr_network_input{nullptr, 0, 0};
		MatrixT attr_network_output{nullptr, 0, 0};
		TempBuffer(NerfNetwork<T>* nerf_network, uint32_t batch_size, cudaStream_t stream, bool output_cm=true) {
			// density
			density_network_input  = MatrixT{nerf_network->m_density_encoding->padded_output_width(), batch_size, 
									         stream, nerf_network->m_density_encoding->preferred_output_layout()};
			density_network_output = MatrixT{nerf_network->m_density_network->padded_output_width(), batch_size,
									         stream, output_cm ? tcnn::MatrixLayout::ColumnMajor : tcnn::MatrixLayout::RowMajor};
			// rgb
			rgb_network_input      = MatrixT{nerf_network->m_density_network->padded_output_width() + nerf_network->m_rgb_encoding->padded_output_width(), batch_size,
									         stream, nerf_network->m_rgb_encoding->preferred_output_layout()};
			rgb_network_output     = MatrixT{nerf_network->m_rgb_network->padded_output_width(), batch_size,
									         stream, output_cm ? tcnn::MatrixLayout::ColumnMajor : tcnn::MatrixLayout::RowMajor};
			// attr
			if (nerf_network->n_attr_dims() > 0) {
				attr_network_input  = MatrixT{nerf_network->m_density_network->padded_output_width() + nerf_network->m_attr_encoding->padded_output_width(), batch_size,
									          stream, nerf_network->m_attr_encoding->preferred_output_layout()};
				attr_network_output = MatrixT{nerf_network->m_attr_network->padded_output_width(), batch_size,
									          stream, output_cm ? tcnn::MatrixLayout::ColumnMajor : tcnn::MatrixLayout::RowMajor};
			}
		}
		void rgb_attr_network_output_zero_(cudaStream_t stream) { // zero out all the elements
			CUDA_CHECK_THROW(cudaMemsetAsync(rgb_network_output.data(), 0, rgb_network_output.n_bytes(), stream));
			if (attr_network_output.data()) {	
				CUDA_CHECK_THROW(cudaMemsetAsync(attr_network_output.data(), 0, attr_network_output.n_bytes(), stream));
			}
		}
	};
	struct ForwardContext : public tcnn::Context, public TempBuffer {
		// density
		std::unique_ptr<tcnn::Context> density_network_ctx = nullptr;
		std::unique_ptr<tcnn::Context> density_encoding_ctx = nullptr;
		// rgb
		std::unique_ptr<tcnn::Context> rgb_network_ctx = nullptr;
		std::unique_ptr<tcnn::Context> rgb_encoding_ctx = nullptr;
		// attr
		std::unique_ptr<tcnn::Context> attr_network_ctx = nullptr;
		std::unique_ptr<tcnn::Context> attr_encoding_ctx = nullptr;
		// init
		using TempBuffer::TempBuffer;
	};
};

NGP_NAMESPACE_END
