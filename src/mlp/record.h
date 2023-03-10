#pragma once

#include <array>
#include <cstdint>

#include "meta.h"

struct Record {
	constexpr static size_t size = 6;
	constexpr static size_t inputs = 5;

	uint16_t year;
	uint8_t month, day;
	FLOAT temperature, wind_speed, solar_radiation, air_pressure, humidity, evaporation;

	const std::array<FLOAT, inputs>& as_input() const {
		return *reinterpret_cast<const std::array<FLOAT, inputs>*>(&temperature);
	}
	NODISCARD FLOAT output() const noexcept {
		return evaporation;
	}
};
