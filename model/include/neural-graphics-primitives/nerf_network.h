/* Doesn't depend on view direction, Has extra rendering attributes
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>

NGP_NAMESPACE_BEGIN

template <typename T>
__global__ void zero_out_dir_grad(uint32_t n_elems, T* __restrict__ ptr, uint32_t m, uint32_t n, bool is_col_major) {
	const uint32_t ij = threadIdx.x + blockIdx.x * blockDim.x;
	if (ij >= n_elems) return;

	auto indexing = [&](uint32_t i, uint32_t j) {
		if (is_col_major) return i + j * m;
		return i * n + j;
	};

	const uint32_t j = (ij / 3u);
	const uint32_t i = (ij - j * 3u) + 4u;
	ptr[indexing(i, j)] = T(0);
}


template <typename T>
class NerfNetwork : public tcnn::Network<float, T> {
public:
	// dimension of extra attributes
	static constexpr uint32_t n_extra_attr = 0u;

	// rgb + density + attr
	static constexpr uint32_t network_output_offset_rgb = 0u;
	static constexpr uint32_t network_output_offset_density = 3u;
	static constexpr uint32_t network_output_offset_attr = 4u;

	using json = nlohmann::json;

	NerfNetwork(const json& encoding, const json& network) {
		uint32_t alignment = (network.contains("otype") && (
				tcnn::equals_case_insensitive(network["otype"], "FullyFusedMLP") || 
				tcnn::equals_case_insensitive(network["otype"], "MegakernelMLP")
			)) ? 16u : 8u ;
		m_encoding.reset(tcnn::create_encoding<T>(3, encoding, alignment));

		json local_network_config = network;
		local_network_config["n_input_dims"] = m_encoding->padded_output_width();
		local_network_config["n_output_dims"] = 4 + n_extra_attr;
		m_network.reset(tcnn::create_network<T>(local_network_config));
	}
	virtual ~NerfNetwork() { }

	void inference_mixed_precision_impl(
		cudaStream_t stream, 
		const tcnn::GPUMatrixDynamic<float>& input, 
		tcnn::GPUMatrixDynamic<T>& output, 
		bool use_inference_params = true
	) override {
		auto input_position = input.slice_rows(0, m_encoding->input_width());
		// runs
		tcnn::GPUMatrixDynamic<T> network_input = {m_encoding->padded_output_width(), input_position.n(), stream, m_encoding->preferred_output_layout()};
		m_encoding->inference_mixed_precision(stream, input_position, network_input, use_inference_params);
		m_network->inference_mixed_precision(stream, network_input, output, use_inference_params);
	}

	uint32_t num_encoded_dims() const {
		return m_encoding->padded_output_width();
	}

	std::unique_ptr<tcnn::Context> forward_impl(
		cudaStream_t stream, 
		const tcnn::GPUMatrixDynamic<float>& input, 
		tcnn::GPUMatrixDynamic<T>* output = nullptr, 
		bool use_inference_params = false, 
		bool prepare_input_gradients = false
	) override {
		auto input_position = input.slice_rows(0, m_encoding->input_width());
		auto forward = std::make_unique<ForwardContext>();
		// runs
		forward->network_input = tcnn::GPUMatrixDynamic<T>{m_encoding->padded_output_width(), input_position.n(), stream, m_encoding->preferred_output_layout()};
		forward->encoding_ctx = m_encoding->forward(stream, input_position, &forward->network_input, use_inference_params, prepare_input_gradients);
		forward->network_ctx = m_network->forward(stream, forward->network_input, output, use_inference_params, prepare_input_gradients);
		return forward;
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
		tcnn::GPUMatrixDynamic<float> input_position = input.slice_rows(0, m_encoding->input_width());
		
		tcnn::GPUMatrixDynamic<float> dL_dinput_position{nullptr, 0, 0};
		if (dL_dinput) {
			dL_dinput_position = dL_dinput->slice_rows(0, m_encoding->input_width());
		}
		
		tcnn::GPUMatrixDynamic<T> dL_dnetwork_input{nullptr, 0, 0};
		if (m_encoding->n_params() > 0 || dL_dinput) {
			dL_dnetwork_input = {m_encoding->padded_output_width(), input.n(), stream, m_encoding->preferred_output_layout()};
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		m_network->backward(stream, *forward.network_ctx, 
			forward.network_input, output, dL_doutput, 
			dL_dnetwork_input.data() ? &dL_dnetwork_input : nullptr, 
			use_inference_params, param_gradients_mode
		);
		if (dL_dnetwork_input.data()) {
			m_encoding->backward(stream, *forward.encoding_ctx, 
				input_position, forward.network_input, dL_dnetwork_input, 
				dL_dinput_position.data() ? &dL_dinput_position : nullptr,
				use_inference_params, param_gradients_mode
			);
		}

		if (dL_dinput) {
			// zero out direction's gradient
			auto m = dL_dinput->m(), n = dL_dinput->n();
			tcnn::linear_kernel(zero_out_dir_grad<float>, 0, stream,
				3 * n, dL_dinput->data(), m, n,
				dL_dinput->layout() == tcnn::MatrixLayout::ColumnMajor
			);
		}
	}

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		size_t offset = 0;
		// setting network
		m_network->set_params(params + offset, inference_params + offset, backward_params + offset, gradients + offset);
		offset += m_network->n_params();
		// setting encoding
		m_encoding->set_params(params + offset, inference_params + offset, backward_params + offset, gradients + offset);
		offset += m_encoding->n_params();
	}

	void initialize_params(
		tcnn::pcg32& rnd, float* params_full_precision, 
		T* params, T* inference_params, T* backward_params, T* gradients, 
		float scale = 1.0f
	) override {
		size_t offset = 0;
		// random initialize network params
		m_network->initialize_params(rnd,
			params_full_precision + offset, params + offset, inference_params + offset, 
			backward_params + offset, gradients + offset,
			scale
		);
		offset += m_network->n_params();
		// random initialize encoding params
		m_encoding->initialize_params(rnd,
			params_full_precision + offset, params + offset, inference_params + offset,
			backward_params + offset, gradients + offset,
			scale
		);
		offset += m_encoding->n_params();
	}

	size_t n_params() const override {
		return m_encoding->n_params() + m_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return m_network->padded_output_width();
	}

	uint32_t output_width() const override {
		return m_network->output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return m_network->layer_sizes();
	}

	uint32_t width(uint32_t layer) const override {
		return layer == 0 ? m_encoding->padded_output_width() : m_network->width(layer - 1);
	}

	uint32_t num_forward_activations() const override {
		return m_network->num_forward_activations() + 1;
	}

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		return layer == 0 ? 
				std::make_pair<const T*, tcnn::MatrixLayout>(forward.network_input.data(), m_encoding->preferred_output_layout()) : 
				m_network->forward_activations(*forward.network_ctx, layer - 1);
	}

	uint32_t input_width() const override {
		// must be the same as `sizeof(NerfCoordinate) / sizeof(float)`
		// pos, dt, dir
		return 3u + 1u + 3u; // although we don't use dir
	}

	const std::shared_ptr<tcnn::Encoding<T>>& encoding() const {
		return m_encoding;
	}

	tcnn::json hyperparams() const override {
		json network_hyperparams = m_network->hyperparams();
		network_hyperparams["n_output_dims"] = m_network->padded_output_width();
		return {
			{"otype", "NerfNetwork"},
			{"encoding", m_encoding->hyperparams()},
			{"network", network_hyperparams}
		};
	}

	//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-// 

	void density(cudaStream_t stream, 
				 const tcnn::GPUMatrixDynamic<float>& input, 
				 tcnn::GPUMatrixDynamic<T>& output, 
				 bool use_inference_params = true) {
		if (input.layout() != tcnn::MatrixLayout::ColumnMajor) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}
		auto input_position = input.slice_rows(0, m_encoding->input_width());
		tcnn::GPUMatrixDynamic<T> network_input = {m_encoding->padded_output_width(), input.n(), stream, m_encoding->preferred_output_layout()};
		m_encoding->inference_mixed_precision(stream, input_position, network_input, use_inference_params);
		m_network->inference_mixed_precision(stream, network_input, output, use_inference_params);
	}

	uint32_t n_extra_dims() const {
		return 0;
	}

	uint32_t padded_density_output_width() const {
		return m_network->padded_output_width();
	}

private:
	std::unique_ptr<tcnn::Network<T>> m_network;
	std::shared_ptr<tcnn::Encoding<T>> m_encoding;

	struct ForwardContext : public tcnn::Context {
		tcnn::GPUMatrixDynamic<T> network_input;
		std::unique_ptr<tcnn::Context> encoding_ctx;
		std::unique_ptr<tcnn::Context> network_ctx;
	};
};

NGP_NAMESPACE_END
