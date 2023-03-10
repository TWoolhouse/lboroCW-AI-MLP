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
			std::ifstream file;
			file.exceptions(std::ios::failbit | std::ios::badbit);
			try {
				file.open(path, std::ifstream::in | std::ifstream::binary);
				Set* sets[] = { &train, &validate, &test };
				for (auto set : sets) {
					size_t size;
					file.read((char*)&size, sizeof(size));
					set->resize(size);
				}
				size_t encoding_size;
				file.read((char*)&encoding_size, sizeof(encoding_size));
				mlp_assert(encoding_size == encodings.size(), "Number of Encodings mismatches the number of columns in the dataset!");

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

				file.close();
				success = true;
			}
			catch (std::ios::failure& e) {
				mlp_log_fatal("Dataset [{}] could not be opened: {}", path.generic_string(), e.what());
			}
			catch (std::exception& e) {
				mlp_log_fatal("Dataset [{}] failed reading: ", path.generic_string(), e.what());
			}
		}

		NODISCARD FLOAT decode_output(FLOAT value) const {
			return encodings.back().decode(value);
		}

	protected:
		bool success = false;
	public:
		NODISCARD operator bool() const { return success; }
	};

}  // namespace mlp
