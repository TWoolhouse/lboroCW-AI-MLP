#include "log/log.h"
#include "meta/define.h"
#include "meta/settings.h"

namespace mlp {}

#define NODISCARD [[nodiscard]]

#define MLP_UNUSED(x) (void)(x)

#ifdef MLP_ASSERT
#define mlp_assert(condition, message, ...) ::mlp::log::assert(condition, std::source_location::current(), mlp_fmt(message, __VA_ARGS__))
#else  // !MLP_ASSERT
#define mlp_assert(condition, message, ...)
#endif	// MLP_ASSERT

#define FLOAT double
