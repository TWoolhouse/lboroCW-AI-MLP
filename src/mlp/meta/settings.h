#pragma once

#include "../config.h"

#ifndef MLP_HEIGHT
#error MLP_HEIGHT is not defined!
#endif

#pragma region MLP_ACTIVATION


#pragma region MLP_ACTIVATION

#if !(defined(MLP_ACTIVATION_SIGMOID) || defined(MLP_ACTIVATION_TANH))
#error No MLP Activation Option Selected! MLP_ACTIVATION<OPT> \
OPT: SIGMOID, TANH
#elif defined(MLP_ACTIVATION_SIGMOID) && defined(MLP_ACTIVATION_TANH)
#error Multiple MLP Activation Options Selected!
#endif // MLP_ACTIVATION

#if defined(MLP_ACTIVATION_SIGMOID)
#define MLP_ACTIVATION ::mlp::Activation::SIGMOID
#elif defined(MLP_ACTIVATION_TANH)
#define MLP_ACTIVATION ::mlp::Activation::TANH
#endif // MLP_ACTIVATION

#pragma endregion
