#pragma once

#include <array>

#include "activation.h"
#include "meta.h"

namespace mlp {

	template<size_t Size, Activation Activator>
	struct Node {
		FLOAT bias;
		std::array<FLOAT, Size> weights;

		static constexpr NODISCARD FLOAT activate(FLOAT value) {
			return mlp::activate<Activator>(value);
		}
		static constexpr NODISCARD FLOAT differential(FLOAT value) {
			return mlp::differential<Activator>(value);
		}

		FLOAT compute(const std::array<FLOAT, Size>& inputs) const {
			FLOAT acc = bias;
			for (auto lhs = weights.cbegin(), rhs = inputs.begin(); lhs < weights.cend(); ++lhs, ++rhs)
				acc += *lhs * *rhs;
			return acc;
		}
	};

}  // namespace mlp
