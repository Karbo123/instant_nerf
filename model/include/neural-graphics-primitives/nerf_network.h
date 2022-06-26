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
void submatrix_row_op(const uint32_t n_rows, const uint32_t batch_size, 
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
		local_rgb_network_config["n_input_dims"] = m_shared_length_1 + m_shared_length_2;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));

		// create attr network
		if (m_n_attr_dims > 0) {
			json local_attr_network_config = attr_network;
			local_attr_network_config["n_input_dims"] = m_shared_length_2 + m_shared_length_3;
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
		auto density_network_input = MatrixT{m_density_encoding->padded_output_width(), batch_size, stream, m_density_encoding->preferred_output_layout()};
		auto shared_space          = MatrixT{m_shared_length_1 + m_shared_length_2 + m_shared_length_3, batch_size, stream, tcnn::MatrixLayout::RowMajor};
		auto rgb_network_output    = MatrixT{m_rgb_network->padded_output_width(), batch_size, stream, output.layout()};
		auto attr_network_output   = MatrixT{m_attr_network->padded_output_width(), batch_size, stream, output.layout()};
		auto final_network_output  = MatrixT{output.data(), this->padded_output_width(), batch_size, output.layout()}; // rename actually

		// density
		m_density_encoding->inference_mixed_precision(stream,
			input.slice_rows(0, m_density_encoding->input_width()), // pos
			density_network_input, // temp for positional encoding
			use_inference_params
		);
		auto density_network_output = shared_space.slice_rows(m_shared_length_1, m_shared_length_2);
		m_density_network->inference_mixed_precision(stream, 
			density_network_input,  // temp for positional encoding
			density_network_output, // tmp=[rgb, DENSITY, attr]
			use_inference_params
		);

		// rgb
		auto rgb_encoding_out = shared_space.slice_rows(0, m_shared_length_1);
		m_rgb_encoding->inference_mixed_precision(stream,
			input.slice_rows(4u, m_rgb_encoding->input_width()), // dir
			rgb_encoding_out, // tmp=[RGB, DENSITY, attr]
			use_inference_params
		);
		m_rgb_network->inference_mixed_precision(stream, 
			shared_space.slice_rows(0, m_shared_length_1 + m_shared_length_2), // tmp=[RGB, DENSITY, attr]
			rgb_network_output,
			use_inference_params
		);

		// if have attribute
		if (m_n_attr_dims > 0) {
			// attr
			auto attr_encoding_out = shared_space.slice_rows(m_shared_length_1 + m_shared_length_2,
															 m_shared_length_3);
			m_attr_encoding->inference_mixed_precision(stream,
				input.slice_rows(4u, m_attr_encoding->input_width()), // dir
				attr_encoding_out, // tmp=[RGB, DENSITY, ATTR]
				use_inference_params
			);
			m_attr_network->inference_mixed_precision(stream, 
				shared_space.slice_rows(m_shared_length_1, 
										m_shared_length_2 + m_shared_length_3), // tmp=[RGB, DENSITY, ATTR], 
				attr_network_output,
				use_inference_params
			);
		}

		// copy rgb from `rgb_network_output` to `final_network_output`
		tcnn::linear_kernel(submatrix_op<CopyOverwrite,T, T>, 0, stream,
			3u * batch_size,
			final_network_output.data(), final_network_output.stride(), final_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
			0u, 0u, 3u, batch_size, // copy to the first row
			rgb_network_output.data(), rgb_network_output.stride(), rgb_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
			0u, 0u,
			T(0)
		);

		// copy density from `shared_space` to `final_network_output`
		tcnn::linear_kernel(submatrix_op<CopyOverwrite,T, T>, 0, stream,
			1u * batch_size,
			final_network_output.data(), final_network_output.stride(), final_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
			3u, 0u, 4u, batch_size, // copy to the third row
			shared_space.data(), shared_space.stride(), shared_space.layout() == tcnn::MatrixLayout::ColumnMajor,
			m_shared_length_1, 0u,
			T(0)
		);

		// copy attr from `attr_network_output` to `final_network_output`
		tcnn::linear_kernel(submatrix_op<CopyOverwrite,T, T>, 0, stream,
			m_n_attr_dims * batch_size,
			final_network_output.data(), final_network_output.stride(), final_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
			4u, 0u, 4u + m_n_attr_dims, batch_size, // copy to the fourth row
			attr_network_output.data(), attr_network_output.stride(), attr_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
			0u, 0u,
			T(0)
		);
	}

	uint32_t padded_density_output_width() const {
		return m_shared_length_2;
	}

	std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream, 
		const MatrixF& input, 
		MatrixT* output = nullptr, 
		bool use_inference_params = false, 
		bool prepare_input_gradients = false
	) override {
		auto batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();
		
		// allocate workspace
		forward->density_network_input = {m_density_encoding->padded_output_width(), batch_size, stream, m_density_encoding->preferred_output_layout()};
		forward->shared_space          = {m_shared_length_1 + m_shared_length_2 + m_shared_length_3, batch_size, stream, tcnn::MatrixLayout::RowMajor};
		if (output) forward->rgb_network_output  = MatrixT{m_rgb_network->padded_output_width(), batch_size, stream, output->layout()};
		if (output) forward->attr_network_output = MatrixT{m_attr_network->padded_output_width(), batch_size, stream, output->layout()};
		
		// if it requires output, final results will be saved to `output->data()`
		MatrixT final_network_output{nullptr, 0, 0};
		if (output) {
			final_network_output = MatrixT{output->data(), this->padded_output_width(), batch_size, output->layout()};
		}

		// // // // // // // // // // // // // // // // // // // // // // // // // // // // 
		// DENSITY

		// encoding position
		forward->density_encoding_ctx = m_density_encoding->forward(stream,
			input.slice_rows(0, m_density_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params, prepare_input_gradients
		);
		// density network
		auto density_network_output = forward->shared_space.slice_rows(m_shared_length_1, m_shared_length_2);
		forward->density_network_ctx = m_density_network->forward(stream, 
			forward->density_network_input, 
			&density_network_output, // tmp=[rgb, DENSITY, attr]
			use_inference_params, prepare_input_gradients
		);

		// // // // // // // // // // // // // // // // // // // // // // // // // // // // 
		// RGB

		// encoding direction for rgb
		auto rgb_encoding_out = forward->shared_space.slice_rows(0, m_shared_length_1);
		forward->rgb_encoding_ctx = m_rgb_encoding->forward(stream,
			input.slice_rows(4u, m_rgb_encoding->input_width()),
			&rgb_encoding_out, // tmp=[RGB, DENSITY, attr]
			use_inference_params, prepare_input_gradients
		);

		// rgb network
		forward->rgb_network_ctx = m_rgb_network->forward(stream, 
			forward->shared_space.slice_rows(0, m_shared_length_1 + m_shared_length_2), 
			output ? &forward->rgb_network_output : nullptr, 
			use_inference_params, prepare_input_gradients
		);

		// // // // // // // // // // // // // // // // // // // // // // // // // // // // 
		// ATTR

		// if have attribute
		if (m_n_attr_dims > 0) {
			// encoding direction for attr
			auto attr_encoding_out = forward->shared_space.slice_rows(
					m_shared_length_1 + m_shared_length_2,
					m_shared_length_3
			);
			forward->attr_encoding_ctx = m_attr_encoding->forward(stream,
				input.slice_rows(4u, m_attr_encoding->input_width()),
				&attr_encoding_out, // tmp=[RGB, DENSITY, ATTR]
				use_inference_params, prepare_input_gradients
			);

			// attr network
			forward->attr_network_ctx = m_attr_network->forward(stream, 
				forward->shared_space.slice_rows(m_shared_length_1,
										         m_shared_length_2 + m_shared_length_3), 
				output ? &forward->attr_network_output : nullptr, 
				use_inference_params, prepare_input_gradients
			);
		}

		if (output) {
			// copy rgb from `rgb_network_output` to `final_network_output`
			tcnn::linear_kernel(submatrix_op<CopyOverwrite,T, T>, 0, stream,
				3u * batch_size,
				final_network_output.data(), final_network_output.stride(), final_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
				0u, 0u, 3u, batch_size, // copy to the first row
				forward->rgb_network_output.data(), forward->rgb_network_output.stride(), forward->rgb_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
				0u, 0u,
				T(0)
			);

			// copy density from `shared_space` to `final_network_output`
			tcnn::linear_kernel(submatrix_op<CopyOverwrite,T, T>, 0, stream,
				1u * batch_size,
				final_network_output.data(), final_network_output.stride(), final_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
				3u, 0u, 4u, batch_size, // copy to the third row
				forward->shared_space.data(), forward->shared_space.stride(), forward->shared_space.layout() == tcnn::MatrixLayout::ColumnMajor,
				m_shared_length_1, 0u,
				T(0)
			);

			// copy attr from `attr_network_output` to `final_network_output`
			tcnn::linear_kernel(submatrix_op<CopyOverwrite,T, T>, 0, stream,
				m_n_attr_dims * batch_size,
				final_network_output.data(), final_network_output.stride(), final_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
				4u, 0u, 4u + m_n_attr_dims, batch_size, // copy to the fourth row
				forward->attr_network_output.data(), forward->attr_network_output.stride(), forward->attr_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
				0u, 0u,
				T(0)
			);
		}

		return forward;
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
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// gradient of tmp
		auto dL_dshared_space = MatrixT{m_shared_length_1 + m_shared_length_2 + m_shared_length_3, batch_size, stream, tcnn::MatrixLayout::RowMajor};

		// // // // // // // // // // // // // // // // // // // // // // // // // // // // 
		// RGB

		// copy rgb gradient to `dL_drgb`
		auto dL_drgb = MatrixT{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream)); // have to zero out the padding region
		tcnn::linear_kernel(submatrix_op<CopyOverwrite, T, T>, 0, stream,
			3u * batch_size,
			dL_drgb.data(), dL_drgb.stride(), dL_drgb.layout() == tcnn::MatrixLayout::ColumnMajor,
			0u, 0u, 3u, batch_size,
			dL_doutput.data(), dL_doutput.stride(), dL_doutput.layout() == tcnn::MatrixLayout::ColumnMajor,
			0u, 0u,
			T(0)
		);

		// backward rgb network
		auto dL_drgb_network_input = dL_dshared_space.slice_rows(0, m_shared_length_1 + m_shared_length_2);
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, 
			forward.shared_space.slice_rows(0, m_shared_length_1 + m_shared_length_2), // in
			forward.rgb_network_output, // out (it matters what the padded region is filled)
			dL_drgb, // dL_doutput (if input is padded, it should be padded with zeros)
			&dL_drgb_network_input, // dL_dinput
			use_inference_params, param_gradients_mode
		);

		// backward rgb encoding
		if (m_rgb_encoding->n_params() > 0 || dL_dinput) {
			MatrixF dL_drgb_encoding_input{nullptr, 0, 0};
			if (dL_dinput) {
				dL_drgb_encoding_input = dL_dinput->slice_rows(4u, m_rgb_encoding->input_width());
			}

			m_rgb_encoding->backward(stream, *forward.rgb_encoding_ctx,
				input.slice_rows(4u, m_rgb_encoding->input_width()), // in
				forward.shared_space.slice_rows(0, m_shared_length_1), // out
				dL_dshared_space.slice_rows(0, m_shared_length_1), // dL_doutput
				dL_dinput ? &dL_drgb_encoding_input : nullptr, // dL_dinput
				use_inference_params, param_gradients_mode
			);
		}

		// // // // // // // // // // // // // // // // // // // // // // // // // // // // 
		// ATTR

		if (m_n_attr_dims > 0) {
			// copy attr gradient to `dL_dattr`
			auto dL_dattr = MatrixT{m_attr_network->padded_output_width(), batch_size, stream};
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_dattr.data(), 0, dL_dattr.n_bytes(), stream)); // have to zero out the padding region
			tcnn::linear_kernel(submatrix_op<CopyOverwrite, T, T>, 0, stream,
				m_n_attr_dims * batch_size,
				dL_dattr.data(), dL_dattr.stride(), dL_dattr.layout() == tcnn::MatrixLayout::ColumnMajor,
				0u, 0u, m_n_attr_dims, batch_size,
				dL_doutput.data(), dL_doutput.stride(), dL_doutput.layout() == tcnn::MatrixLayout::ColumnMajor,
				4u, 0u,
				T(0)
			);

			// back up density network gradient from the rgb network former result
			auto dL_ddensity_network_output = MatrixT{m_shared_length_2, batch_size, stream, dL_dshared_space.layout()};
			tcnn::linear_kernel(submatrix_op<CopyOverwrite, T, T>, 0, stream,
				m_shared_length_2 * batch_size,
				dL_ddensity_network_output.data(), dL_ddensity_network_output.stride(), dL_ddensity_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
				0u, 0u, m_shared_length_2, batch_size,
				dL_dshared_space.data(), dL_dshared_space.stride(), dL_dshared_space.layout() == tcnn::MatrixLayout::ColumnMajor,
				m_shared_length_1, 0u,
				T(0)
			);

			// backward attr network
			auto dL_dattr_network_input = dL_dshared_space.slice_rows(m_shared_length_1, 
																	  m_shared_length_2 + m_shared_length_3);
			m_attr_network->backward(stream, *forward.attr_network_ctx, 
				forward.shared_space.slice_rows(m_shared_length_1, 
											    m_shared_length_2 + m_shared_length_3), // in
				forward.attr_network_output, // out (it matters what the padded region is filled)
				dL_dattr, // dL_doutput (if input is padded, it should be padded with zeros)
				&dL_dattr_network_input, // dL_dinput
				use_inference_params, param_gradients_mode // because it overwrites the gradient buffer, so we need to make backup (i.e. dL_ddensity_first_row)
			);

			// accumulate density gradient from the rgb network
			tcnn::linear_kernel(submatrix_op<CopyAccumulate, T, T>, 0, stream,
				m_shared_length_2 * batch_size,
				dL_dshared_space.data(), dL_dshared_space.stride(), dL_dshared_space.layout() == tcnn::MatrixLayout::ColumnMajor,
				m_shared_length_1, 0u, m_shared_length_1 + m_shared_length_2, batch_size,
				dL_ddensity_network_output.data(), dL_ddensity_network_output.stride(), dL_ddensity_network_output.layout() == tcnn::MatrixLayout::ColumnMajor,
				0u, 0u,
				T(0)
			);

			// backward attr encoding
			if (m_attr_encoding->n_params() > 0 || dL_dinput) {

				// back up direction gradient from the rgb network
				auto dL_ddir = MatrixT{3, batch_size, stream, dL_dinput->layout()};
				tcnn::linear_kernel(submatrix_op<CopyOverwrite, T, float>, 0, stream,
					3u * batch_size,
					dL_ddir.data(), dL_ddir.stride(), dL_ddir.layout() == tcnn::MatrixLayout::ColumnMajor,
					0u, 0u, 3u, batch_size,
					dL_dinput->data(), dL_dinput->stride(), dL_dinput->layout() == tcnn::MatrixLayout::ColumnMajor,
					4u, 0u,
					T(0)
				);

				MatrixF dL_dattr_encoding_input{nullptr, 0, 0};
				if (dL_dinput) {
					dL_dattr_encoding_input = dL_dinput->slice_rows(4u, m_attr_encoding->input_width());
				}

				m_attr_encoding->backward(stream, *forward.attr_encoding_ctx,
					input.slice_rows(4u, m_attr_encoding->input_width()), // in
					forward.shared_space.slice_rows(m_shared_length_1 + m_shared_length_2, 
													m_shared_length_3), // out
					dL_dshared_space.slice_rows(m_shared_length_1 + m_shared_length_2, m_shared_length_3), // dL_doutput
					dL_dinput ? &dL_dattr_encoding_input : nullptr, // dL_dinput
					use_inference_params, param_gradients_mode // because it overwrites the gradient buffer, so we need to make backup (i.e. dL_ddir)
				);

				// accumulate direction gradient from the rgb network
				tcnn::linear_kernel(submatrix_op<CopyAccumulate, float, T>, 0, stream,
					3u * batch_size,
					dL_dinput->data(), dL_dinput->stride(), dL_dinput->layout() == tcnn::MatrixLayout::ColumnMajor,
					4u, 0u, 7u, batch_size,
					dL_ddir.data(), dL_ddir.stride(), dL_ddir.layout() == tcnn::MatrixLayout::ColumnMajor,
					0u, 0u,
					T(0)
				);
			}
		}

		// // // // // // // // // // // // // // // // // // // // // // // // // // // // 
		// DENSITY

		// accumulate density gradient from `dL_doutput` to `dL_dshared_space`
		tcnn::linear_kernel(submatrix_op<CopyAccumulate, T, T>, 0, stream,
			batch_size,
			dL_dshared_space.data(), dL_dshared_space.stride(), dL_dshared_space.layout() == tcnn::MatrixLayout::ColumnMajor,
			m_shared_length_1, 0u, 1u + m_shared_length_1, batch_size, 
			dL_doutput.data(), dL_doutput.stride(), dL_doutput.layout() == tcnn::MatrixLayout::ColumnMajor,
			3u, 0u, // the third row is density gradient
			T(0)
		);

		// backward density network
		MatrixT dL_ddensity_network_input;
		if (m_density_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = MatrixT{m_density_encoding->padded_output_width(), batch_size, stream, m_density_encoding->preferred_output_layout()};
		}
		m_density_network->backward(stream, *forward.density_network_ctx, 
			forward.density_network_input, // in
			forward.shared_space.slice_rows(m_shared_length_1, m_shared_length_2), // out
			dL_dshared_space.slice_rows(m_shared_length_1, m_shared_length_2), 
			dL_dinput ? &dL_ddensity_network_input : nullptr, 
			use_inference_params, param_gradients_mode
		);

		// backward density encoding
		if (dL_dinput) {
			MatrixF dL_dpos_encoding_input{nullptr, 0, 0};
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_density_encoding->input_width());
			}

			m_density_encoding->backward(stream, *forward.density_encoding_ctx,
				input.slice_rows(0, m_density_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
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
			input.slice_rows(0, m_density_encoding->input_width()),
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
		return 4 + m_n_attr_dims;
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
		density_network_hyperparams["n_output_dims"] = m_shared_length_2;
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

	// necessaries to compute gradient
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
	};
};

NGP_NAMESPACE_END
