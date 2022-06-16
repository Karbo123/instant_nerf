/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/tinyexr_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <json/json.hpp>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <fstream>
#include <set>

// Windows.h is evil
#undef min
#undef max
#undef near
#undef far


using namespace Eigen;
using namespace std::literals::chrono_literals;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

std::atomic<size_t> g_total_n_bytes_allocated{0};

json merge_parent_network_config(const json &child, const fs::path &child_filename) {
	if (!child.contains("parent")) {
		return child;
	}
	fs::path parent_filename = child_filename.parent_path() / std::string(child["parent"]);
	tlog::info() << "Loading parent network config from: " << parent_filename.str();
	std::ifstream f{parent_filename.str()};
	json parent = json::parse(f, nullptr, true, true);
	parent = merge_parent_network_config(parent, parent_filename);
	parent.merge_patch(child);
	return parent;
}

static bool ends_with(const std::string& str, const std::string& ending) {
	if (ending.length() > str.length()) {
		return false;
	}
	return std::equal(std::rbegin(ending), std::rend(ending), std::rbegin(str));
}

void Testbed::load_training_data(const std::string& data_path) {
	m_data_path = data_path;

	if (!m_data_path.exists()) {
		throw std::runtime_error{std::string{"Data path '"} + m_data_path.str() + "' does not exist."};
	}

	switch (m_testbed_mode) {
		case ETestbedMode::Nerf:  load_nerf(); break;
		default: throw std::runtime_error{"Invalid testbed mode."};
	}

	m_training_data_available = true;
}

void Testbed::clear_training_data() {
	m_training_data_available = false;
	m_nerf.training.dataset.metadata.clear();
}

json Testbed::load_network_config(const fs::path& network_config_path) {
	if (!network_config_path.empty()) {
		m_network_config_path = network_config_path;
	}

	tlog::info() << "Loading network config from: " << network_config_path;

	if (network_config_path.empty() || !network_config_path.exists()) {
		throw std::runtime_error{std::string{"Network config \""} + network_config_path.str() + "\" does not exist."};
	}

	json result;
	if (equals_case_insensitive(network_config_path.extension(), "json")) {
		std::ifstream f{network_config_path.str()};
		result = json::parse(f, nullptr, true, true);
		result = merge_parent_network_config(result, network_config_path);
	} else if (equals_case_insensitive(network_config_path.extension(), "msgpack")) {
		std::ifstream f{network_config_path.str(), std::ios::in | std::ios::binary};
		result = json::from_msgpack(f);
		// we assume parent pointers are already resolved in snapshots.
	}

	return result;
}

void Testbed::reload_network_from_file(const std::string& network_config_path) {
	if (!network_config_path.empty()) {
		m_network_config_path = network_config_path;
	}

	m_network_config = load_network_config(m_network_config_path);
	reset_network();
}

void Testbed::reload_network_from_json(const json& json, const std::string& config_base_path) {
	// config_base_path is needed so that if the passed in json uses the 'parent' feature, we know where to look...
	// be sure to use a filename, or if a directory, end with a trailing slash
	m_network_config = merge_parent_network_config(json, config_base_path);
	reset_network();
}

void Testbed::handle_file(const std::string& file) {
	if (ends_with(file, ".msgpack")) {
		load_snapshot(file);
	}
	else if (ends_with(file, ".json")) {
		reload_network_from_file(file);
	} else {
		tlog::error() << "Tried to open unknown file type: " << file;
	}
}

void Testbed::reset_accumulation(bool due_to_camera_movement, bool immediate_redraw) {
	if (immediate_redraw) {
		redraw_next_frame();
	}

	if (!due_to_camera_movement || !reprojection_available()) {
		m_windowless_render_surface.reset_accumulation();
		for (auto& tex : m_render_surfaces) {
			tex.reset_accumulation();
		}
	}
}

void Testbed::set_visualized_dim(int dim) {
	m_visualized_dimension = dim;
	reset_accumulation();
}

void Testbed::translate_camera(const Vector3f& rel) {
	m_camera.col(3) += m_camera.block<3,3>(0,0) * rel * m_bounding_radius;
	reset_accumulation(true);
}

void Testbed::set_nerf_camera_matrix(const Matrix<float, 3, 4>& cam) {
	m_camera = m_nerf.training.dataset.nerf_matrix_to_ngp(cam);
}

Vector3f Testbed::look_at() const {
	return view_pos() + view_dir() * m_scale;
}

void Testbed::set_look_at(const Vector3f& pos) {
	m_camera.col(3) += pos - look_at();
}

void Testbed::set_scale(float scale) {
	auto prev_look_at = look_at();
	m_camera.col(3) = (view_pos() - prev_look_at) * (scale / m_scale) + prev_look_at;
	m_scale = scale;
}

void Testbed::set_view_dir(const Vector3f& dir) {
	auto old_look_at = look_at();
	m_camera.col(0) = dir.cross(m_up_dir).normalized();
	m_camera.col(1) = dir.cross(m_camera.col(0)).normalized();
	m_camera.col(2) = dir.normalized();
	set_look_at(old_look_at);
}

void Testbed::set_camera_to_training_view(int trainview) {
	auto old_look_at = look_at();
	m_camera = m_smoothed_camera = m_nerf.training.transforms[trainview].start;
	m_relative_focal_length = m_nerf.training.dataset.metadata[trainview].focal_length / (float)m_nerf.training.dataset.metadata[trainview].resolution[m_fov_axis];
	m_scale = std::max((old_look_at - view_pos()).dot(view_dir()), 0.1f);
	m_nerf.render_with_camera_distortion = true;
	m_nerf.render_distortion = m_nerf.training.dataset.metadata[trainview].camera_distortion;
	m_screen_center = Vector2f::Constant(1.0f) - m_nerf.training.dataset.metadata[0].principal_point;
}

void Testbed::reset_camera() {
	m_fov_axis = 1;
	set_fov(50.625f);
	m_zoom = 1.f;
	m_screen_center = Vector2f::Constant(0.5f);
	m_scale = m_testbed_mode == ETestbedMode::Image ? 1.0f : 1.5f;
	m_camera <<
		1.0f, 0.0f, 0.0f, 0.5f,
		0.0f, -1.0f, 0.0f, 0.5f,
		0.0f, 0.0f, -1.0f, 0.5f;
	m_camera.col(3) -= m_scale * view_dir();
	m_smoothed_camera = m_camera;
	m_up_dir = {0.0f, 1.0f, 0.0f};
	m_sun_dir = Vector3f::Ones().normalized();
	reset_accumulation();
}

void Testbed::set_train(bool mtrain) {
	if (m_train && !mtrain && m_max_level_rand_training) {
		set_max_level(1.f);
	}
	m_train = mtrain;
}

std::string get_filename_in_data_path_with_suffix(fs::path data_path, fs::path network_config_path, const char* suffix) {
	// use the network config name along with the data path to build a filename with the requested suffix & extension
	std::string default_name = network_config_path.basename();
	if (default_name == "") default_name = "base";
	if (data_path.empty())
		return default_name + std::string(suffix);
	if (data_path.is_directory())
		return (data_path / (default_name + std::string{suffix})).str();
	else
		return data_path.stem().str() + "_" + default_name + std::string(suffix);
}

void Testbed::compute_and_save_marching_cubes_mesh(const char* filename, Vector3i res3d , BoundingBox aabb, float thresh, bool unwrap_it) {
	if (aabb.is_empty()) {
		aabb = m_testbed_mode == ETestbedMode::Nerf ? m_render_aabb : m_aabb;
	}
	marching_cubes(res3d, aabb, thresh);
	save_mesh(m_mesh.verts, m_mesh.vert_normals, m_mesh.vert_colors, m_mesh.indices, filename, unwrap_it, m_nerf.training.dataset.scale, m_nerf.training.dataset.offset);
}

Eigen::Vector3i Testbed::compute_and_save_png_slices(const char* filename, int res, BoundingBox aabb, float thresh, float density_range, bool flip_y_and_z_axes) {
	if (aabb.is_empty()) {
		aabb = m_testbed_mode == ETestbedMode::Nerf ? m_render_aabb : m_aabb;
	}
	if (thresh == std::numeric_limits<float>::max()) {
		thresh = m_mesh.thresh;
	}
	float range = density_range;
	if (m_testbed_mode == ETestbedMode::Sdf) {
		auto res3d = get_marching_cubes_res(res, aabb);
		aabb.inflate(range * aabb.diag().x()/res3d.x());
	}
	auto res3d = get_marching_cubes_res(res, aabb);
	if (m_testbed_mode == ETestbedMode::Sdf)
		range *= -aabb.diag().x()/res3d.x(); // rescale the range to be in output voxels. ie this scale factor is mapped back to the original world space distances.
			// negated so that black = outside, white = inside
	char fname[128];
	snprintf(fname, sizeof(fname), ".density_slices_%dx%dx%d.png", res3d.x(), res3d.y(), res3d.z());
	GPUMemory<float> density = get_density_on_grid(res3d, aabb);
	save_density_grid_to_png(density, (std::string(filename) + fname).c_str(), res3d, thresh, flip_y_and_z_axes, range);
	return res3d;
}

inline float linear_to_db(float x) {
	return -10.f*logf(x)/logf(10.f);
}

void Testbed::dump_parameters_as_images() {
	size_t non_layer_params_width = 2048;

	size_t layer_params = 0;
	for (auto size : m_network->layer_sizes()) {
		layer_params += size.first * size.second;
	}

	size_t non_layer_params = m_network->n_params() - layer_params;

	float* params = m_trainer->params();
	std::vector<float> params_cpu(layer_params + next_multiple(non_layer_params, non_layer_params_width), 0.0f);
	CUDA_CHECK_THROW(cudaMemcpy(params_cpu.data(), params, m_network->n_params() * sizeof(float), cudaMemcpyDeviceToHost));

	size_t offset = 0;
	size_t layer_id = 0;
	for (auto size : m_network->layer_sizes()) {
		std::string filename = std::string{"layer-"} + std::to_string(layer_id) + ".exr";
		save_exr(params_cpu.data() + offset, size.second, size.first, 1, 1, filename.c_str());
		offset += size.first * size.second;
		++layer_id;
	}

	std::string filename = "non-layer.exr";
	save_exr(params_cpu.data() + offset, non_layer_params_width, non_layer_params / non_layer_params_width, 1, 1, filename.c_str());
}

void Testbed::train_and_render(bool skip_rendering) {
	if (m_train) {
		train(m_training_batch_size);
	}

	if (m_mesh.optimize_mesh) {
		optimise_mesh_step(1);
	}

	apply_camera_smoothing(m_frame_ms.val());
}

bool Testbed::frame() {
	// Render against the trained neural network. If we're training and already close to convergence,
	// we can skip rendering if the scene camera doesn't change
	uint32_t n_to_skip = m_train ? tcnn::clamp(m_training_step / 16u, 15u, 255u) : 0;
	if (m_render_skip_due_to_lack_of_camera_movement_counter > n_to_skip) {
		m_render_skip_due_to_lack_of_camera_movement_counter = 0;
	}
	bool skip_rendering = m_render_skip_due_to_lack_of_camera_movement_counter++ != 0;
	if (!skip_rendering || (std::chrono::steady_clock::now() - m_last_gui_draw_time_point) > 100ms) {
		redraw_gui_next_frame();
	}

	try {
		while (true) {
			(*m_task_queue.tryPop())();
		}
	} catch (SharedQueueEmptyException&) {}


	train_and_render(skip_rendering);

	return true;
}

fs::path Testbed::training_data_path() const {
	return m_data_path.with_extension("training");
}

bool Testbed::want_repl() {
	bool b=m_want_repl;
	m_want_repl=false;
	return b;
}

void Testbed::apply_camera_smoothing(float elapsed_ms) {
	if (m_camera_smoothing) {
		float decay = std::pow(0.02f, elapsed_ms/1000.0f);
		m_smoothed_camera = log_space_lerp(m_smoothed_camera, m_camera, 1.0f - decay);
	} else {
		m_smoothed_camera = m_camera;
	}
}

CameraKeyframe Testbed::copy_camera_to_keyframe() const {
	return CameraKeyframe(m_camera, m_slice_plane_z, m_scale, fov(), m_dof );
}

void Testbed::set_camera_from_keyframe(const CameraKeyframe& k) {
	m_camera = k.m();
	m_slice_plane_z = k.slice;
	m_scale = k.scale;
	set_fov(k.fov);
	m_dof = k.dof;
}

void Testbed::set_camera_from_time(float t) {
	if (m_camera_path.m_keyframes.empty())
		return;
	set_camera_from_keyframe(m_camera_path.eval_camera_path(t));
}

void Testbed::update_loss_graph() {
	m_loss_graph[m_loss_graph_samples++ % m_loss_graph.size()] = std::log(m_loss_scalar.val());
}

uint32_t Testbed::n_dimensions_to_visualize() const {
	return m_network->width(m_visualized_layer);
}

float Testbed::fov() const {
	return focal_length_to_fov(1.0f, m_relative_focal_length[m_fov_axis]);
}

void Testbed::set_fov(float val) {
	m_relative_focal_length = Vector2f::Constant(fov_to_focal_length(1, val));
}

Vector2f Testbed::fov_xy() const {
	return focal_length_to_fov(Vector2i::Ones(), m_relative_focal_length);
}

void Testbed::set_fov_xy(const Vector2f& val) {
	m_relative_focal_length = fov_to_focal_length(Vector2i::Ones(), val);
}

size_t Testbed::n_params() {
	return m_network->n_params();
}

size_t Testbed::n_encoding_params() {
	return m_network->n_params() - first_encoder_param();
}

size_t Testbed::first_encoder_param() {
	auto layer_sizes = m_network->layer_sizes();
	size_t first_encoder = 0;
	for (auto size : layer_sizes) {
		first_encoder += size.first * size.second;
	}
	return first_encoder;
}

uint32_t Testbed::network_width(uint32_t layer) const {
	return m_network->width(layer);
}

uint32_t Testbed::network_num_forward_activations() const {
	return m_network->num_forward_activations();
}

void Testbed::set_max_level(float maxlevel) {
	if (!m_network) return;
	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_max_level(maxlevel);
	}
	reset_accumulation();
}

void Testbed::set_min_level(float minlevel) {
	if (!m_network) return;
	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_quantize_threshold(powf(minlevel, 4.f) * 0.2f);
	}
	reset_accumulation();
}

void Testbed::set_visualized_layer(int layer) {
	m_visualized_layer = layer;
	m_visualized_dimension = std::max(-1, std::min(m_visualized_dimension, (int)m_network->width(layer)-1));
	reset_accumulation();
}

ELossType Testbed::string_to_loss_type(const std::string& str) {
	if (equals_case_insensitive(str, "L2")) {
		return ELossType::L2;
	} else if (equals_case_insensitive(str, "RelativeL2")) {
		return ELossType::RelativeL2;
	} else if (equals_case_insensitive(str, "L1")) {
		return ELossType::L1;
	} else if (equals_case_insensitive(str, "Mape")) {
		return ELossType::Mape;
	} else if (equals_case_insensitive(str, "Smape")) {
		return ELossType::Smape;
	} else if (equals_case_insensitive(str, "Huber") || equals_case_insensitive(str, "SmoothL1")) {
		// Legacy: we used to refer to the Huber loss (L2 near zero, L1 further away) as "SmoothL1".
		return ELossType::Huber;
	} else if (equals_case_insensitive(str, "LogL1")) {
		return ELossType::LogL1;
	} else {
		throw std::runtime_error{"Unknown loss type."};
	}
}

Testbed::NetworkDims Testbed::network_dims() const {
	switch (m_testbed_mode) {
		case ETestbedMode::Nerf:   return network_dims_nerf(); break;
		default: throw std::runtime_error{"Invalid mode."};
	}
}

void Testbed::reset_network() {
	m_sdf.iou_decay = 0;

	m_rng = default_rng_t{m_seed};

	// Start with a low rendering resolution and gradually ramp up
	m_render_ms.set(10000);

	reset_accumulation();
	m_nerf.training.counters_rgb.rays_per_batch = 1 << 12;
	m_nerf.training.counters_rgb.measured_batch_size_before_compaction = 0;
	m_nerf.training.n_steps_since_cam_update = 0;
	m_nerf.training.n_steps_since_error_map_update = 0;
	m_nerf.training.n_rays_since_error_map_update = 0;
	m_nerf.training.n_steps_between_error_map_updates = 128;
	m_nerf.training.error_map.is_cdf_valid = false;

	m_nerf.training.reset_camera_extrinsics();

	m_loss_graph_samples = 0;

	// Default config
	json config = m_network_config;

	json& encoding_config = config["encoding"];
	json& loss_config = config["loss"];
	json& optimizer_config = config["optimizer"];
	json& network_config = config["network"];

	auto dims = network_dims();

	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf.training.loss_type = string_to_loss_type(loss_config.value("otype", "L2"));

		// Some of the Nerf-supported losses are not supported by tcnn::Loss,
		// so just create a dummy L2 loss there. The NeRF code path will bypass
		// the tcnn::Loss in any case.
		loss_config["otype"] = "L2";
	}

	// Automatically determine certain parameters if we're dealing with the (hash)grid encoding
	if (to_lower(encoding_config.value("otype", "OneBlob")).find("grid") != std::string::npos) {
		encoding_config["n_pos_dims"] = dims.n_pos;

		const uint32_t n_features_per_level = encoding_config.value("n_features_per_level", 2u);

		if (encoding_config.contains("n_features") && encoding_config["n_features"] > 0) {
			m_num_levels = (uint32_t)encoding_config["n_features"] / n_features_per_level;
		} else {
			m_num_levels = encoding_config.value("n_levels", 16u);
		}

		m_level_stats.resize(m_num_levels);
		m_first_layer_column_stats.resize(m_num_levels);

		const uint32_t log2_hashmap_size = encoding_config.value("log2_hashmap_size", 15);

		m_base_grid_resolution = encoding_config.value("base_resolution", 0);
		if (!m_base_grid_resolution) {
			m_base_grid_resolution = 1u << ((log2_hashmap_size) / dims.n_pos);
			encoding_config["base_resolution"] = m_base_grid_resolution;
		}

		float desired_resolution = 2048.0f; // Desired resolution of the finest hashgrid level over the unit cube
		if (m_testbed_mode == ETestbedMode::Image) {
			desired_resolution = m_image.resolution.maxCoeff() / 2.0f;
		} else if (m_testbed_mode == ETestbedMode::Volume) {
			desired_resolution = m_volume.world2index_scale;
		}

		// Automatically determine suitable per_level_scale
		m_per_level_scale = encoding_config.value("per_level_scale", 0.0f);
		if (m_per_level_scale <= 0.0f && m_num_levels > 1) {
			m_per_level_scale = std::exp(std::log(desired_resolution * (float)m_nerf.training.dataset.aabb_scale / (float)m_base_grid_resolution) / (m_num_levels-1));
			encoding_config["per_level_scale"] = m_per_level_scale;
		}

		tlog::info()
			<< "GridEncoding: "
			<< " Nmin=" << m_base_grid_resolution
			<< " b=" << m_per_level_scale
			<< " F=" << n_features_per_level
			<< " T=2^" << log2_hashmap_size
			<< " L=" << m_num_levels
			;
	}

	m_loss.reset(create_loss<precision_t>(loss_config));
	m_optimizer.reset(create_optimizer<precision_t>(optimizer_config));

	size_t n_encoding_params = 0;
	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Array3f>(1e-3f, Array3f::Zero()));
		m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<Vector3f>(1e-4f, Vector3f::Zero()));
		m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
		m_nerf.training.cam_focal_length_offset = AdamOptimizer<Vector2f>(1e-5f);

		m_nerf.training.reset_extra_dims(m_rng);

		json& dir_encoding_config = config["dir_encoding"];
		json& rgb_network_config = config["rgb_network"];

		uint32_t n_dir_dims = 3;
		uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
		m_network = m_nerf_network = std::make_shared<NerfNetwork<precision_t>>(
			dims.n_pos,
			n_dir_dims,
			n_extra_dims,
			dims.n_pos + 1, // The offset of 1 comes from the dt member variable of NerfCoordinate. HACKY
			encoding_config,
			dir_encoding_config,
			network_config,
			rgb_network_config
		);

		m_encoding = m_nerf_network->encoding();
		n_encoding_params = m_encoding->n_params() + m_nerf_network->dir_encoding()->n_params();

		tlog::info()
			<< "Density model: " << dims.n_pos
			<< "--[" << std::string(encoding_config["otype"])
			<< "]-->" << m_nerf_network->encoding()->padded_output_width()
			<< "--[" << std::string(network_config["otype"])
			<< "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"]+2) << ")"
			<< "]-->" << 1
			;

		tlog::info()
			<< "Color model:   " << n_dir_dims
			<< "--[" << std::string(dir_encoding_config["otype"])
			<< "]-->" << m_nerf_network->dir_encoding()->padded_output_width() << "+" << network_config.value("n_output_dims", 16u)
			<< "--[" << std::string(rgb_network_config["otype"])
			<< "(neurons=" << (int)rgb_network_config["n_neurons"] << ",layers=" << ((int)rgb_network_config["n_hidden_layers"]+2) << ")"
			<< "]-->" << 3
			;

		// Create distortion map model
		{
			json& distortion_map_optimizer_config =  config.contains("distortion_map") && config["distortion_map"].contains("optimizer") ? config["distortion_map"]["optimizer"] : optimizer_config;

			m_distortion.resolution = Vector2i::Constant(32);
			if (config.contains("distortion_map") && config["distortion_map"].contains("resolution")) {
				from_json(config["distortion_map"]["resolution"], m_distortion.resolution);
			}
			m_distortion.map = std::make_shared<TrainableBuffer<2, 2, float>>(m_distortion.resolution);
			m_distortion.optimizer.reset(create_optimizer<float>(distortion_map_optimizer_config));
			m_distortion.trainer = std::make_shared<Trainer<float, float>>(m_distortion.map, m_distortion.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(loss_config)}, m_seed);
		}
	} else {
		uint32_t alignment = network_config.contains("otype") && (equals_case_insensitive(network_config["otype"], "FullyFusedMLP") || equals_case_insensitive(network_config["otype"], "MegakernelMLP")) ? 16u : 8u;

		if (encoding_config.contains("otype") && equals_case_insensitive(encoding_config["otype"], "Takikawa")) {
			if (m_sdf.octree_depth_target == 0) {
				m_sdf.octree_depth_target = encoding_config["n_levels"];
			}

			if (!m_sdf.triangle_octree || m_sdf.triangle_octree->depth() != m_sdf.octree_depth_target) {
				m_sdf.triangle_octree.reset(new TriangleOctree{});
				m_sdf.triangle_octree->build(*m_sdf.triangle_bvh, m_sdf.triangles_cpu, m_sdf.octree_depth_target);
				m_sdf.octree_depth_target = m_sdf.triangle_octree->depth();
				m_sdf.brick_data.free_memory();
			}

			m_encoding.reset(new TakikawaEncoding<precision_t>(
				encoding_config["starting_level"],
				m_sdf.triangle_octree,
				tcnn::string_to_interpolation_type(encoding_config.value("interpolation", "linear"))
			));

			m_network = std::make_shared<NetworkWithInputEncoding<precision_t>>(m_encoding, dims.n_output, network_config);
			m_sdf.uses_takikawa_encoding = true;
		} else {
			m_encoding.reset(create_encoding<precision_t>(dims.n_input, encoding_config));
			m_network = std::make_shared<NetworkWithInputEncoding<precision_t>>(m_encoding, dims.n_output, network_config);
			m_sdf.uses_takikawa_encoding = false;
			if (m_sdf.octree_depth_target == 0 && encoding_config.contains("n_levels")) {
				m_sdf.octree_depth_target = encoding_config["n_levels"];
			}
		}

		n_encoding_params = m_encoding->n_params();

		tlog::info()
			<< "Model:         " << dims.n_input
			<< "--[" << std::string(encoding_config["otype"])
			<< "]-->" << m_encoding->padded_output_width()
			<< "--[" << std::string(network_config["otype"])
			<< "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"]+2) << ")"
			<< "]-->" << dims.n_output;
	}

	size_t n_network_params = m_network->n_params() - n_encoding_params;

	tlog::info() << "  total_encoding_params=" << n_encoding_params << " total_network_params=" << n_network_params;

	m_trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(m_network, m_optimizer, m_loss, m_seed);
	m_training_step = 0;
	m_training_start_time_point = std::chrono::steady_clock::now();

	// Create envmap model
	{
		json& envmap_loss_config = config.contains("envmap") && config["envmap"].contains("loss") ? config["envmap"]["loss"] : loss_config;
		json& envmap_optimizer_config =  config.contains("envmap") && config["envmap"].contains("optimizer") ? config["envmap"]["optimizer"] : optimizer_config;

		m_envmap.loss_type = string_to_loss_type(envmap_loss_config.value("otype", "L2"));

		m_envmap.resolution = m_nerf.training.dataset.envmap_resolution;
		m_envmap.envmap = std::make_shared<TrainableBuffer<4, 2, float>>(m_envmap.resolution);
		m_envmap.optimizer.reset(create_optimizer<float>(envmap_optimizer_config));
		m_envmap.trainer = std::make_shared<Trainer<float, float, float>>(m_envmap.envmap, m_envmap.optimizer, std::shared_ptr<Loss<float>>{create_loss<float>(envmap_loss_config)}, m_seed);

		if (m_nerf.training.dataset.envmap_data.data()) {
			m_envmap.trainer->set_params_full_precision(m_nerf.training.dataset.envmap_data.data(), m_nerf.training.dataset.envmap_data.size());
		}
	}
}

Testbed::Testbed(ETestbedMode mode)
: m_testbed_mode(mode)
{
	uint32_t compute_capability = cuda_compute_capability();
	if (compute_capability < MIN_GPU_ARCH) {
		tlog::warning() << "Insufficient compute capability " << compute_capability << " detected.";
		tlog::warning() << "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly.";
	}

	m_network_config = {
		{"loss", {
			{"otype", "L2"}
		}},
		{"optimizer", {
			{"otype", "Adam"},
			{"learning_rate", 1e-3},
			{"beta1", 0.9f},
			{"beta2", 0.99f},
			{"epsilon", 1e-15f},
			{"l2_reg", 1e-6f},
		}},
		{"encoding", {
			{"otype", "HashGrid"},
			{"n_levels", 16},
			{"n_features_per_level", 2},
			{"log2_hashmap_size", 19},
			{"base_resolution", 16},
		}},
		{"network", {
			{"otype", "FullyFusedMLP"},
			{"n_neurons", 64},
			{"n_layers", 2},
			{"activation", "ReLU"},
			{"output_activation", "None"},
		}},
	};

	reset_camera();

	if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
		throw std::runtime_error{"Testbed required CUDA 10.2 or later."};
	}

	set_exposure(0);
	set_min_level(0.f);
	set_max_level(1.f);

	CUDA_CHECK_THROW(cudaStreamCreate(&m_inference_stream));
	m_training_stream = m_inference_stream;
}

Testbed::~Testbed() {}

void Testbed::train(uint32_t batch_size) {
	if (!m_training_data_available) {
		m_train = false;
		return;
	}

	if (!m_dlss) {
		// No immediate redraw necessary
		reset_accumulation(false, false);
	}

	uint32_t n_prep_to_skip = m_testbed_mode == ETestbedMode::Nerf ? tcnn::clamp(m_training_step / 16u, 1u, 16u) : 1u;
	if (m_training_step % n_prep_to_skip == 0) {
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_prep_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count() / n_prep_to_skip);
		}};

		switch (m_testbed_mode) {
			case ETestbedMode::Nerf:   training_prep_nerf(batch_size, m_training_stream);  break;
			default: throw std::runtime_error{"Invalid training mode."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_training_stream));
	}

	// Find leaf optimizer and update its settings
	json* leaf_optimizer_config = &m_network_config["optimizer"];
	while (leaf_optimizer_config->contains("nested")) {
		leaf_optimizer_config = &(*leaf_optimizer_config)["nested"];
	}
	(*leaf_optimizer_config)["optimize_matrix_params"] = m_train_network;
	(*leaf_optimizer_config)["optimize_non_matrix_params"] = m_train_encoding;
	m_optimizer->update_hyperparams(m_network_config["optimizer"]);

	bool get_loss_scalar = m_training_step % 16 == 0;

	{
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count());
		}};

		switch (m_testbed_mode) {
			case ETestbedMode::Nerf:   train_nerf(batch_size, get_loss_scalar, m_training_stream); break;
			default: throw std::runtime_error{"Invalid training mode."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_training_stream));
	}

	if (get_loss_scalar) {
		update_loss_graph();
	}
}

Vector2f Testbed::calc_focal_length(const Vector2i& resolution, int fov_axis, float zoom) const {
	return m_relative_focal_length * resolution[fov_axis] * zoom;
}

Vector2f Testbed::render_screen_center() const {
	// see pixel_to_ray for how screen center is used; 0.5,0.5 is 'normal'. we flip so that it becomes the point in the original image we want to center on.
	auto screen_center = m_screen_center;
	return {(0.5f-screen_center.x())*m_zoom + 0.5f, (0.5-screen_center.y())*m_zoom + 0.5f};
}

__global__ void dlss_prep_kernel(
	ETestbedMode mode,
	Vector2i resolution,
	uint32_t sample_index,
	Vector2f focal_length,
	Vector2f screen_center,
	Vector3f parallax_shift,
	bool snap_to_pixel_centers,
	float* depth_buffer,
	Matrix<float, 3, 4> camera,
	Matrix<float, 3, 4> prev_camera,
	cudaSurfaceObject_t depth_surface,
	cudaSurfaceObject_t mvec_surface,
	cudaSurfaceObject_t exposure_surface,
	CameraDistortion camera_distortion,
	const float view_dist,
	const float prev_view_dist,
	const Vector2f image_pos,
	const Vector2f prev_image_pos,
	const Vector2i image_resolution
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	uint32_t x_orig = x;
	uint32_t y_orig = y;


	const float depth = depth_buffer[idx];
	Vector2f mvec = mode == ETestbedMode::Image ? motion_vector_2d(
		sample_index,
		{x, y},
		resolution,
		image_resolution,
		screen_center,
		view_dist,
		prev_view_dist,
		image_pos,
		prev_image_pos,
		snap_to_pixel_centers
	) : motion_vector_3d(
		sample_index,
		{x, y},
		resolution,
		focal_length,
		camera,
		prev_camera,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		depth,
		camera_distortion
	);

	surf2Dwrite(make_float2(mvec.x(), mvec.y()), mvec_surface, x_orig * sizeof(float2), y_orig);

	// Scale depth buffer to be guaranteed in [0,1].
	surf2Dwrite(std::min(std::max(depth / 128.0f, 0.0f), 1.0f), depth_surface, x_orig * sizeof(float), y_orig);

	// First thread write an exposure factor of 1. Since DLSS will run on tonemapped data,
	// exposure is assumed to already have been applied to DLSS' inputs.
	if (x_orig == 0 && y_orig == 0) {
		surf2Dwrite(1.0f, exposure_surface, 0, 0);
	}
}

void Testbed::render_frame(const Matrix<float, 3, 4>& camera_matrix0, const Matrix<float, 3, 4>& camera_matrix1, const Vector4f& nerf_rolling_shutter, CudaRenderBuffer& render_buffer, bool to_srgb) {
	Vector2i max_res = m_window_res.cwiseMax(render_buffer.in_resolution());

	render_buffer.clear_frame(m_inference_stream);

	Vector2f focal_length = calc_focal_length(render_buffer.in_resolution(), m_fov_axis, m_zoom);
	Vector2f screen_center = render_screen_center();

	switch (m_testbed_mode) {
		case ETestbedMode::Nerf:
			if (!m_render_ground_truth) {
				render_nerf(render_buffer, max_res, focal_length, camera_matrix0, camera_matrix1, nerf_rolling_shutter, screen_center, m_inference_stream);
			}
			break;
		default:
			throw std::runtime_error{"Invalid render mode."};
	}

	render_buffer.set_color_space(m_color_space);
	render_buffer.set_tonemap_curve(m_tonemap_curve);

	// Prepare DLSS data: motion vectors, scaled depth, exposure
	if (render_buffer.dlss()) {
		auto res = render_buffer.in_resolution();

		bool distortion = m_testbed_mode == ETestbedMode::Nerf && m_nerf.render_with_camera_distortion;

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };

		Vector3f parallax_shift = get_scaled_parallax_shift();
		if (parallax_shift.head<2>() != Vector2f::Zero()) {
			throw std::runtime_error{"Motion vectors don't support parallax shift."};
		}

		dlss_prep_kernel<<<blocks, threads, 0, m_inference_stream>>>(
			m_testbed_mode,
			res,
			render_buffer.spp(),
			focal_length,
			screen_center,
			parallax_shift,
			m_snap_to_pixel_centers,
			render_buffer.depth_buffer(),
			camera_matrix0,
			m_prev_camera,
			render_buffer.dlss()->depth(),
			render_buffer.dlss()->mvec(),
			render_buffer.dlss()->exposure(),
			distortion ? m_nerf.render_distortion : CameraDistortion{},
			m_scale,
			m_prev_scale,
			m_image.pos,
			m_image.prev_pos,
			m_image.resolution
		);

		render_buffer.set_dlss_sharpening(m_dlss_sharpening);
	}

	m_prev_camera = camera_matrix0;
	m_prev_scale = m_scale;
	m_image.prev_pos = m_image.pos;

	render_buffer.accumulate(m_exposure, m_inference_stream);
	render_buffer.tonemap(m_exposure, m_background_color, to_srgb ? EColorSpace::SRGB : EColorSpace::Linear, m_inference_stream);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		// Overlay the ground truth image if requested
		if (m_render_ground_truth) {
			float alpha=1.f;
			auto const &metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
			render_buffer.overlay_image(
				alpha,
				Array3f::Constant(m_exposure) + m_nerf.training.cam_exposure[m_nerf.training.view].variable(),
				m_background_color,
				to_srgb ? EColorSpace::SRGB : EColorSpace::Linear,
				metadata.pixels,
				metadata.image_data_type,
				metadata.resolution,
				m_fov_axis,
				m_zoom,
				Vector2f::Constant(0.5f),
				m_inference_stream
			);
		}

		// Visualize the accumulated error map if requested
		if (m_nerf.training.render_error_overlay) {
			const float* err_data = m_nerf.training.error_map.data.data();
			Vector2i error_map_res = m_nerf.training.error_map.resolution;
			if (m_render_ground_truth) {
				err_data = m_nerf.training.dataset.sharpness_data.data();
				error_map_res = m_nerf.training.dataset.sharpness_resolution;
			}
			size_t emap_size = error_map_res.x() * error_map_res.y();
			err_data += emap_size * m_nerf.training.view;
			static GPUMemory<float> average_error;
			average_error.enlarge(1);
			average_error.memset(0);
			const float* aligned_err_data_s = (const float*)(((size_t)err_data)&~15);
			const float* aligned_err_data_e = (const float*)(((size_t)(err_data+emap_size))&~15);
			size_t reduce_size = aligned_err_data_e - aligned_err_data_s;
			reduce_sum(aligned_err_data_s, [reduce_size] __device__ (float val) { return max(val,0.f) / (reduce_size); }, average_error.data(), reduce_size, m_inference_stream);
			auto const &metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
			render_buffer.overlay_false_color(metadata.resolution, to_srgb, m_fov_axis, m_inference_stream, err_data, error_map_res, average_error.data(), m_nerf.training.error_overlay_brightness, m_render_ground_truth);
		}
	}

	CUDA_CHECK_THROW(cudaStreamSynchronize(m_inference_stream));
}

void Testbed::determine_autofocus_target_from_pixel(const Vector2i& focus_pixel) {
	float depth;

	const auto& surface = m_render_surfaces.front();
	if (surface.depth_buffer()) {
		auto res = surface.in_resolution();
		Vector2i depth_pixel = focus_pixel.cast<float>().cwiseProduct(res.cast<float>()).cwiseQuotient(m_window_res.cast<float>()).cast<int>();
		depth_pixel = depth_pixel.cwiseMin(res).cwiseMax(0);

		CUDA_CHECK_THROW(cudaMemcpy(&depth, surface.depth_buffer() + depth_pixel.x() + depth_pixel.y() * res.x(), sizeof(float), cudaMemcpyDeviceToHost));
	} else {
		depth = m_scale;
	}

	auto ray = pixel_to_ray_pinhole(0, focus_pixel, m_window_res, calc_focal_length(m_window_res, m_fov_axis, m_zoom), m_smoothed_camera, render_screen_center());

	m_autofocus_target = ray.o + ray.d * depth;
	m_autofocus = true; // If someone shift-clicked, that means they want the AUTOFOCUS
}

void Testbed::autofocus() {
	float new_slice_plane_z = std::max(view_dir().dot(m_autofocus_target - view_pos()), 0.1f) - m_scale;
	if (new_slice_plane_z != m_slice_plane_z) {
		m_slice_plane_z = new_slice_plane_z;
		if (m_dof != 0.0f) {
			reset_accumulation();
		}
	}
}

Testbed::LevelStats compute_level_stats(const float* params, size_t n_params) {
	Testbed::LevelStats s = {};
	for (size_t i = 0; i < n_params; ++i) {
		float v = params[i];
		float av = fabsf(v);
		if (av < 0.00001f) {
			s.numzero++;
		} else {
			if (s.count == 0) s.min = s.max = v;
			s.count++;
			s.x += v;
			s.xsquared += v * v;
			s.min = min(s.min, v);
			s.max = max(s.max, v);
		}
	}
	return s;
}

void Testbed::gather_histograms() {
	int n_params = (int)m_network->n_params();
	int first_encoder = first_encoder_param();
	int n_encoding_params = n_params - first_encoder;

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc && m_trainer->params()) {
		std::vector<float> grid(n_encoding_params);

		uint32_t m = m_network->layer_sizes().front().first;
		uint32_t n = m_network->layer_sizes().front().second;
		std::vector<float> first_layer_rm(m * n);

		CUDA_CHECK_THROW(cudaMemcpyAsync(grid.data(), m_trainer->params() + first_encoder, grid.size() * sizeof(float), cudaMemcpyDeviceToHost, m_training_stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(first_layer_rm.data(), m_trainer->params(), first_layer_rm.size() * sizeof(float), cudaMemcpyDeviceToHost, m_training_stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(m_training_stream));


		for (int l = 0; l < m_num_levels; ++l) {
			m_level_stats[l] = compute_level_stats(grid.data() + hg_enc->level_params_offset(l), hg_enc->level_n_params(l));
		}

		int numquant = 0;
		m_quant_percent = float(numquant * 100) / (float)n_encoding_params;
		if (m_histo_level < m_num_levels) {
			size_t nperlevel = hg_enc->level_n_params(m_histo_level);
			const float* d = grid.data() + hg_enc->level_params_offset(m_histo_level);
			float scale = 128.f / (m_histo_scale); // fixed scale for now to make it more comparable between levels
			memset(m_histo, 0, sizeof(m_histo));
			for (int i = 0; i < nperlevel; ++i) {
				float v = *d++;
				if (v == 0.f) {
					continue;
				}
				int bin = (int)floor(v * scale + 128.5f);
				if (bin >= 0 && bin <= 256) {
					m_histo[bin]++;
				}
			}
		}
	}
}

void Testbed::save_snapshot(const std::string& filepath_string, bool include_optimizer_state) {
	fs::path filepath = filepath_string;
	m_network_config["snapshot"] = m_trainer->serialize(include_optimizer_state);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_network_config["snapshot"]["density_grid_size"] = NERF_GRIDSIZE();
		m_network_config["snapshot"]["density_grid_binary"] = m_nerf.density_grid;
	}

	m_network_config["snapshot"]["training_step"] = m_training_step;
	m_network_config["snapshot"]["loss"] = m_loss_scalar.val();

	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_network_config["snapshot"]["nerf"]["rgb"]["rays_per_batch"] = m_nerf.training.counters_rgb.rays_per_batch;
		m_network_config["snapshot"]["nerf"]["rgb"]["measured_batch_size"] = m_nerf.training.counters_rgb.measured_batch_size;
		m_network_config["snapshot"]["nerf"]["rgb"]["measured_batch_size_before_compaction"] = m_nerf.training.counters_rgb.measured_batch_size_before_compaction;
		m_network_config["snapshot"]["nerf"]["dataset"] = m_nerf.training.dataset;
	}

	m_network_config_path = filepath;
	std::ofstream f(m_network_config_path.str(), std::ios::out | std::ios::binary);
	json::to_msgpack(m_network_config, f);
}

void Testbed::load_snapshot(const std::string& filepath_string) {
	auto config = load_network_config(filepath_string);
	if (!config.contains("snapshot")) {
		throw std::runtime_error{std::string{"File '"} + filepath_string + "' does not contain a snapshot."};
	}

	m_network_config_path = filepath_string;
	m_network_config = config;

	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf.training.counters_rgb.rays_per_batch = m_network_config["snapshot"]["nerf"]["rgb"]["rays_per_batch"];
		m_nerf.training.counters_rgb.measured_batch_size = m_network_config["snapshot"]["nerf"]["rgb"]["measured_batch_size"];
		m_nerf.training.counters_rgb.measured_batch_size_before_compaction = m_network_config["snapshot"]["nerf"]["rgb"]["measured_batch_size_before_compaction"];
		// If we haven't got a nerf dataset loaded, load dataset metadata from the snapshot
		// and render using just that.
		if (m_data_path.empty() && m_network_config["snapshot"]["nerf"].contains("dataset")) {
			m_nerf.training.dataset = m_network_config["snapshot"]["nerf"]["dataset"];
			load_nerf();
		}

		if (m_network_config["snapshot"]["density_grid_size"] != NERF_GRIDSIZE()) {
			throw std::runtime_error{"Incompatible grid size in snapshot."};
		}

		m_nerf.density_grid = m_network_config["snapshot"]["density_grid_binary"];
		update_density_grid_mean_and_bitfield(nullptr);
	}

	reset_network();

	m_training_step = m_network_config["snapshot"]["training_step"];
	m_loss_scalar.set(m_network_config["snapshot"]["loss"]);

	m_trainer->deserialize(m_network_config["snapshot"]);
}

void Testbed::load_camera_path(const std::string& filepath_string) {
	m_camera_path.load(filepath_string, Matrix<float, 3, 4>::Identity());
}

NGP_NAMESPACE_END

