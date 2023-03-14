#pragma once

#include "meta.h"
#include "node.h"
#include "model.h"
#include "activation.h"

#define MLP_TRAIN_MOMENTUM 0.9
// #define MLP_TRAIN_BOlD_DRIVER
// #define MLP_TRAIN_ANNEALING
// #define MLP_TRAIN_WEIGHT_DECAY


namespace mlp {

	template<size_t Inputs, Activation Activator, size_t Height>
	class Trainer {
		FLOAT learning_rate = 0.005;
		#ifdef MLP_TRAIN_MOMENTUM
		static constexpr FLOAT momentum_weight = MLP_TRAIN_MOMENTUM;
		#endif // MLP_TRAIN_MOMENTUM

		template<size_t Size>
		struct Cell {
			Node<Size, Activator> node;
			union {
				FLOAT sum, delta;
			};
			FLOAT activated;

			#ifdef MLP_TRAIN_MOMENTUM
			struct Momentum {
				FLOAT bias;
				std::array<FLOAT, Size> weights;

				constexpr FLOAT& remap(FLOAT& node_ptr) {
					auto ptr = reinterpret_cast<uint8_t*>(&node_ptr);
					return *reinterpret_cast<FLOAT*>(ptr + offset());
				}
			protected:
				static constexpr auto offset() {
					return offsetof(Cell, momentum) - offsetof(Cell, node);
				}
			} momentum;
			#endif // MLP_TRAIN_MOMENTUM


			// Forward pass of a single Node
			auto forward(const std::array<FLOAT, Size>& inputs) {
				sum = node.compute(inputs);
				activated = node.activate(sum);
				return activated;
			}

			static constexpr FLOAT backward_error(FLOAT correct, FLOAT guess) {
				#ifdef MLP_TRAIN_WEIGHT_DECAY
				#else // !MLP_TRAIN_WEIGHT_DECAY
				return correct - guess;
				#endif // MLP_TRAIN_WEIGHT_DECAY
			}

			void backward_delta_output(FLOAT correct) {
				delta = backward_error(correct, activated) * node.differential(sum);
			}
			void backward_delta_hidden(FLOAT fwd_delta, FLOAT fwd_weight) {
				delta = fwd_weight * fwd_delta * node.differential(sum);
			}
		};
	protected:
		std::array<Cell<Inputs>, Height> layer;
		Cell<Height> output;

	public:
		// Forward & Backward Pass
		FLOAT train(const std::array<FLOAT, Inputs>& inputs, FLOAT correct) {
			auto guess = forward(inputs);
			backward(inputs, correct);
			return correct - guess;
		}

		FLOAT bold_driver() {
			// TODO: Edit learning rate
		}

	protected:
		// Forwards pass
		FLOAT forward(const std::array<FLOAT, Inputs>& inputs) {
			auto hidden = forward_hidden(inputs);
			return output.forward(hidden);
		}

		// Forward pass for the hidden layer
		std::array<FLOAT, Height> forward_hidden(const std::array<FLOAT, Inputs>& inputs) {
			std::array<FLOAT, Height> outputs;
			auto it = outputs.data();
			for (auto& cell : layer) {
				*it++ = cell.forward(inputs);
			}
			return outputs;
		}

		void backward(const std::array<FLOAT, Inputs>& inputs, FLOAT correct) {
			output.backward_delta_output(correct);
			auto it_weight_out = output.node.weights.rbegin();
			for (auto it_cell = layer.rbegin(); it_cell < layer.rend(); ++it_cell) {
				it_cell->backward_delta_hidden(output.delta, *it_weight_out);
				backward_update_bias(it_cell->node.bias, *it_cell);
				auto it_weight_node = it_cell->node.weights.begin();
				for (auto it_input = inputs.begin(); it_input < inputs.end(); ++it_input)
					backward_update_weight(*it_weight_node++, *it_cell, *it_input);
				backward_update_weight(*it_weight_out++, output, it_cell->activated);
			}
			backward_update_bias(output.node.bias, output);
		}

		template<size_t Size>
		void backward_update_weight(FLOAT& weight, Cell<Size>& cell, FLOAT input) {
			#ifdef MLP_TRAIN_MOMENTUM
			auto old = weight;
			auto& momentum = cell.momentum.remap(weight);
			weight += learning_rate * cell.delta * input + momentum_weight * momentum;
			momentum = weight - old;
			#else // !MLP_TRAIN_MOMENTUM
			weight += learning_rate * cell.delta * input;
			#endif // MLP_TRAIN_MOMENTUM
		}

		template<size_t Size>
		auto backward_update_bias(FLOAT& bias, Cell<Size>& cell) {
			return backward_update_weight(bias, cell, 1);
		}

	public:
		Model<Inputs, Activator, Height> model() const {
			std::array<Node<Inputs, Activator>, Height> nodes;
			for (size_t i = 0; i < Height; ++i) {
				nodes[i] = layer[i].node;
			}
			return Model<Inputs, Activator, Height>{ std::move(nodes), output.node };
		}

		Trainer(const Model<Inputs, Activator, Height>& model): layer(), output() {
			for (size_t i = 0; i < Height; ++i) {
				layer[i].node = model.layer[i];
			}
			output.node = model.output;
		}
	};

} // namespace mlp
