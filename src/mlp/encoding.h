#pragma once

#include <cstdint>
#include "meta.h"

namespace mlp {

	struct Encoding {
	public:
		FLOAT min, max;
		FLOAT lower, upper;
	protected:
		FLOAT range_set, range_bounds;

	public:
		constexpr Encoding() = default;
		constexpr Encoding(FLOAT min, FLOAT max, FLOAT lower, FLOAT upper): min(min), max(max), lower(lower), upper(upper), range_set(max - min), range_bounds(upper - lower) {}

		constexpr NODISCARD FLOAT encode(FLOAT value) const {
			auto normalise = (value - min) / range_set;
			return normalise * range_bounds + lower;
		}
		constexpr NODISCARD FLOAT decode(FLOAT value) const {
			auto normalise = (value - lower) / range_bounds;
			return normalise * range_set + min;
		}
	};

} // namespace mlp
