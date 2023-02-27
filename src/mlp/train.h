#pragma once

#include "meta.h"
#include "node.h"
#include "model.h"
#include "activation.h"

constexpr FLOAT learning_rate = 0.005;

namespace mlp {

	template<size_t Inputs, Activation Activator, size_t Height>
	class TrainModel {
		template<size_t Size>
		struct Cell {
			Node<Size, Activator> node;
			union {
				FLOAT sum, delta;
			};
			FLOAT activated;

			auto forward(const std::array<FLOAT, Size>& inputs) {
				sum = node.compute(inputs);
				activated = node.activate(sum);
				return activated;
			}
			void backward(FLOAT weight, FLOAT up_delta) {
				delta = weight * up_delta * node.differential(sum);
			}
		};
	protected:
		std::array<Cell<Inputs>, Height> layer;
		Cell<Height> output;

	public:
		FLOAT train(const std::array<FLOAT, Inputs>& inputs, FLOAT correct) {
			auto guess = forward(inputs);
			backward(inputs, guess, correct);
			return guess - correct;
		}

	protected:
		FLOAT forward(const std::array<FLOAT, Inputs>& inputs) {
			auto hidden = forward_hidden(inputs);
			return output.forward(hidden);
		}

		std::array<FLOAT, Height> forward_hidden(const std::array<FLOAT, Inputs>& inputs) {
			std::array<FLOAT, Height> outputs;
			auto out_it = outputs.data();
			for (auto& cell : layer) {
				*out_it++ = cell.forward(inputs);
			}
			return outputs;
		}

		void backward(const std::array<FLOAT, Inputs>& inputs, FLOAT guess, FLOAT correct) {
			output.delta = (correct - guess) * output.node.differential(output.sum);
			auto it_weight = output.node.weights.rbegin();
			for (auto it = layer.rbegin(); it < layer.rend(); ++it) {
				it->backward(*it_weight, output.delta);
				it->node.bias += learning_rate * it->delta;
				auto node_weight = it->node.weights.begin();
				for (auto input = inputs.begin(); input < inputs.end(); ++input)
					*(node_weight++) += learning_rate * it->delta * *input;
				*(it_weight++) += learning_rate * output.delta * it->activated;
			}
		}

	public:
		Model<Inputs, Activator, Height> model() const {
			std::array<Node<Inputs, Activator>, Height> nodes;
			for (size_t i = 0; i < Height; ++i) {
				nodes[i] = layer[i].node;
			}
			return Model<Inputs, Activator, Height>{ std::move(nodes), output.node };
		}
	};

} // namespace mlp
