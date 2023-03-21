#pragma once

#include "../config.h"

#ifndef MLP_HEIGHT
#error MLP_HEIGHT is not defined!
#endif

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

#pragma region MLP_TRAIN_BOLD_DRIVER

#ifdef MLP_TRAIN_BOLD_DRIVER
#define _MC_CONCAT(x, y) x##y
#define MC_CONCAT(x, y) _MC_CONCAT(x, y)

#define _EXTRACT_TUP_0(zero, ...) zero
#define _EXTRACT_TUP_1(zero, one, ...) one
#define _EXTRACT_TUP_2(zero, one, two, ...) two
#define _EXTRACT_TUP_3(zero, one, two, three, ...) three

#define MLP_TRAIN_BOLD_DRIVER_MIN MC_CONCAT(_EXTRACT_TUP_0, MLP_TRAIN_BOLD_DRIVER)
#define MLP_TRAIN_BOLD_DRIVER_MAX MC_CONCAT(_EXTRACT_TUP_1, MLP_TRAIN_BOLD_DRIVER)
#define MLP_TRAIN_BOLD_DRIVER_INC MC_CONCAT(_EXTRACT_TUP_2, MLP_TRAIN_BOLD_DRIVER)
#define MLP_TRAIN_BOLD_DRIVER_DEC MC_CONCAT(_EXTRACT_TUP_3, MLP_TRAIN_BOLD_DRIVER)
#endif // MLP_TRAIN_BOLD_DRIVER

#pragma endregion

#pragma region MLP_TRAIN_ANNEALING

#ifdef MLP_TRAIN_ANNEALING
#define _MC_CONCAT(x, y) x##y
#define MC_CONCAT(x, y) _MC_CONCAT(x, y)

#define _EXTRACT_TUP_0(zero, ...) zero
#define _EXTRACT_TUP_1(zero, one, ...) one

#define MLP_TRAIN_ANNEALING_START MC_CONCAT(_EXTRACT_TUP_0, MLP_TRAIN_ANNEALING)
#define MLP_TRAIN_ANNEALING_END   MC_CONCAT(_EXTRACT_TUP_1, MLP_TRAIN_ANNEALING)
#endif // MLP_TRAIN_ANNEALING

#pragma endregion
