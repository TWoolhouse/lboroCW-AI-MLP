#pragma once

#include <array>
#include <cstdint>
#include <filesystem>

#include "meta.h"
#include "node.h"

namespace mlp {

	template<size_t Inputs, Activation Activator, size_t Height>
	class Model {
	protected:
		std::array<Node<Inputs, Activator>, Height> layer;
		Node<Height, Activator> output;

	public:
		Model(std::filesystem::path path) {}
		Model(std::array<Node<Inputs, Activator>, Height>&& nodes, Node<Height, Activator> output): layer(std::move(nodes)), output(output) {}

		// Calculate output directly. Without Training!
		FLOAT compute(const std::array<FLOAT, Inputs>& inputs) const {
			auto hidden = compute_hidden(inputs);
			return output.activate(output.compute(hidden));
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
