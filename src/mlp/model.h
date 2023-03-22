#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <random>

#include "meta.h"
#include "node.h"
#include "dataset.h"

namespace mlp {

	template<size_t Inputs, Activation Activator, size_t Height>
	class Trainer;

	template<size_t Inputs, Activation Activator, size_t Height>
	class Model {
		friend class Trainer<Inputs, Activator, Height>;
	protected:
		std::array<Node<Inputs, Activator>, Height> layer;
		Node<Height, Activator> output;

	public:
		Model(): layer(), output() {}
		Model(unsigned int random_seed): layer(), output() {
			std::default_random_engine engine{ random_seed };
			std::uniform_real_distribution<FLOAT> distribution{-static_cast<FLOAT>(Inputs) / 2, static_cast<FLOAT>(Inputs) / 2};
			auto rand = [&]() { return distribution(engine); };
			auto randomise = [&](auto&& node) {
				node.bias = rand();
				for (auto& weight : node.weights)
					weight = rand();
			};

			for (auto& node : layer)
				randomise(node);
			randomise(output);
		}
		Model(std::filesystem::path path): Model() {
			try {
				std::ifstream file;
				file.exceptions(std::ios::badbit | std::ios::failbit);
				file.open(path, std::ios::in | std::ios::binary);
				file.read(reinterpret_cast<char*>(this), sizeof(Model));
				file.close();
			}
			catch (std::ios::failure& e) {
				mlp_log_fatal("Unable to deserialise Model {} : {}", path.generic_string(), e.what());
				throw e;
			}
		}
		Model(std::array<Node<Inputs, Activator>, Height>&& nodes, Node<Height, Activator> output): layer(std::move(nodes)), output(output) {}

		NODISCARD FLOAT compute(const std::array<FLOAT, Inputs>& inputs) const {
			auto hidden = compute_hidden(inputs);
			return output.activate(output.compute(hidden));
		}

		NODISCARD FLOAT forward(const Dataset::Set& dataset) const {
			FLOAT acc = 0;
			for (auto& record : dataset) {
				auto guess = compute(record.as_input());
				auto err = guess - record.output();
				acc += err * err;
			}
			return acc / dataset.size();
		}

	protected:
		auto compute_hidden(const std::array<FLOAT, Inputs>& inputs) const {
			std::array<FLOAT, Height> outputs;
			auto out_it = outputs.data();
			for (const auto& node : layer)
				*out_it++ = node.activate(node.compute(inputs));
			return outputs;
		}
	};

}  // namespace mlp
