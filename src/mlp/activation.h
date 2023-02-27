#pragma once

#include <cstdint>

#include "meta.h"

namespace mlp {
	enum class Activation: uint8_t {
		LINEAR,
		SIGMOID,
		TANH,
	};

	template<Activation Activator>
	constexpr NODISCARD FLOAT activate(FLOAT value) {
		if constexpr (Activator == Activation::LINEAR) {
			return value;
		}
		else if constexpr (Activator == Activation::SIGMOID) {
			return 1.0 / (1.0 + exp(-value));
		}
		else if constexpr (Activator == Activation::TANH) {
			auto epx = exp(value);
			auto enx = exp(-value);
			return (epx - enx) / (epx + enx);
		}
	}

	template<Activation Activator>
	constexpr NODISCARD FLOAT differential(FLOAT value) {
		if constexpr (Activator == Activation::LINEAR) {
			return 1;
		}
		else if constexpr (Activator == Activation::SIGMOID) {
			return activate<Activation::SIGMOID>(value) * (1.0 - activate<Activation::SIGMOID>(value));
		}
		else if constexpr (Activator == Activation::TANH) {
			auto tanh = activate<Activation::TANH>(value);
			return 1.0 - (tanh * tanh);
		}

	}

}  // namespace mlp
