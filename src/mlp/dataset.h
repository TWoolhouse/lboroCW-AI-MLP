#pragma once

#include <filesystem>
#include <fstream>
#include <vector>

#include "record.h"
#include "encoding.h"

namespace mlp {

	struct Dataset {
		using Set = std::vector<Record>;
		std::array<Encoding, Record::size> encodings;
		Set train, validate, test;

		Dataset(std::filesystem::path path) {
			std::ifstream file{path, std::ifstream::in | std::ifstream::binary};
			if (file.is_open()) {
				Set* sets[] = { &train, &validate, &test };
				for (auto set : sets) {
					size_t size;
					file.read((char*)&size, sizeof(size));
					set->resize(size);
				}

				struct EncodingRaw {
					FLOAT min, max;
					FLOAT lower, upper;
				} raw_encoding;
				for (auto& encoding : encodings) {
					file.read((char*)&raw_encoding, sizeof(EncodingRaw));
					encoding = Encoding(raw_encoding.min, raw_encoding.max, raw_encoding.lower, raw_encoding.upper);
				}

				for (auto set : sets)
					file.read((char*)set->data(), sizeof(Set::value_type) * set->size());
			}
		}

		NODISCARD FLOAT decode_output(FLOAT value) const {
			return encodings.back().decode(value);
		}
	};

}  // namespace mlp
