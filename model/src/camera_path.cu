/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   camera_path.cpp
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/camera_path.h>
#include <neural-graphics-primitives/common.h>

#include <json/json.hpp>
#include <fstream>

using namespace Eigen;
using namespace nlohmann;

NGP_NAMESPACE_BEGIN

CameraKeyframe lerp(const CameraKeyframe& p0, const CameraKeyframe& p1, float t, float t0, float t1) {
	t = (t - t0) / (t1 - t0);
	Eigen::Vector4f R1 = p1.R;

	// take the short path
	if (R1.dot(p0.R) < 0.f)  {
		R1=-R1;
	}

	return {
		Eigen::Quaternionf(p0.R).slerp(t, Eigen::Quaternionf(R1)).coeffs(),
		p0.T + (p1.T - p0.T) * t,
		p0.slice + (p1.slice - p0.slice) * t,
		p0.scale + (p1.scale - p0.scale) * t,
		p0.fov + (p1.fov - p0.fov) * t,
		p0.dof + (p1.dof - p0.dof) * t,
	};
}

CameraKeyframe spline(float t, const CameraKeyframe& p0, const CameraKeyframe& p1, const CameraKeyframe& p2, const CameraKeyframe& p3) {
	if (0) {
		// catmull rom spline
		CameraKeyframe q0 = lerp(p0, p1, t, -1.f, 0.f);
		CameraKeyframe q1 = lerp(p1, p2, t,  0.f, 1.f);
		CameraKeyframe q2 = lerp(p2, p3, t,  1.f, 2.f);
		CameraKeyframe r0 = lerp(q0, q1, t, -1.f, 1.f);
		CameraKeyframe r1 = lerp(q1, q2, t,  0.f, 2.f);
		return lerp(r0, r1, t, 0.f, 1.f);
	} else {
		// cublic bspline
		float tt=t*t;
		float ttt=t*t*t;
		float a = (1-t)*(1-t)*(1-t)*(1.f/6.f);
		float b = (3.f*ttt-6.f*tt+4.f)*(1.f/6.f);
		float c = (-3.f*ttt+3.f*tt+3.f*t+1.f)*(1.f/6.f);
		float d = ttt*(1.f/6.f);
		return p0 * a + p1 * b + p2 * c + p3 * d;
	}
}

void to_json(json& j, const CameraKeyframe& p) {
	j = json{{"R", p.R}, {"T", p.T}, {"slice", p.slice}, {"scale", p.scale}, {"fov", p.fov}, {"dof", p.dof}};
}

bool load_relative_to_first=false; // set to true when using a camera path that is aligned with the first training image, such that it is invariant to changes in the space of the training data

void from_json(bool is_first, const json& j, CameraKeyframe& p, const CameraKeyframe& first, const Eigen::Matrix<float, 3, 4>& ref) {
	 if (is_first && load_relative_to_first) {
	 	p.from_m(ref);
	 } else {
		p.R=Eigen::Vector4f(j["R"][0],j["R"][1],j["R"][2],j["R"][3]);
		p.T=Eigen::Vector3f(j["T"][0],j["T"][1],j["T"][2]);

		if (load_relative_to_first) {
	 		Eigen::Matrix4f ref4 = Eigen::Matrix4f::Identity();
	 		ref4.block<3,4>(0,0) = ref;

	 		Eigen::Matrix4f first4 = Eigen::Matrix4f::Identity();
	 		first4.block<3,4>(0,0) = first.m();

	 		Eigen::Matrix4f p4 = Eigen::Matrix4f::Identity();
	 		p4.block<3,4>(0,0) = p.m();

	 		auto cur4 = ref4 * first4.inverse() * p4;
	 		p.from_m(cur4.block<3,4>(0,0));
		}
	}
	j.at("slice").get_to(p.slice);
	j.at("scale").get_to(p.scale);
	j.at("fov").get_to(p.fov);
	j.at("dof").get_to(p.dof);
}


void CameraPath::save(const std::string& filepath_string) {
	json j = {
		{"time", m_playtime},
		{"path", m_keyframes}
	};
	std::ofstream f(filepath_string);
	f << j;
}

void CameraPath::load(const std::string& filepath_string, const Eigen::Matrix<float, 3, 4> &first_xform) {
	std::ifstream f(filepath_string);
	if (!f) {
		throw std::runtime_error{std::string{"Camera path \""} + filepath_string + "\" does not exist."};
	}

	json j;
	f >> j;

	CameraKeyframe first;

	m_keyframes.clear();
	if (j.contains("time")) m_playtime=j["time"];
	if (j.contains("path")) for (auto &el : j["path"]) {
		CameraKeyframe p;
		bool is_first = m_keyframes.empty();
		from_json(is_first, el, p, first, first_xform);
		if (is_first) {
			first = p;
		}
		m_keyframes.push_back(p);
	}
}

NGP_NAMESPACE_END
