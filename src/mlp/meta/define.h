#pragma once

#include "../config.h"

namespace mlp::def {

#ifdef DEBUG
#define MLP_DEBUG
#endif	// DEBUG

//-----------------------------------------------------------------//
//---------------------------MLP_ASSERT----------------------------//
//-----------------------------------------------------------------//

#pragma region MLP_ASSERT

#if !defined(MLP_ASSERT)
#define MLP_ASSERT
#elif MLP_ASSERT == 0
#undef MLP_ASSERT
#endif	// MLP_ASSERT

constexpr bool assert =
#ifdef MLP_ASSERT
	true
#else	// !MLP_ASSERT
	false
#endif	// MLP_ASSERT
	;
#pragma endregion

namespace log {

//-----------------------------------------------------------------//
//-------------------------MLP_LOG_CONSOLE-------------------------//
//-----------------------------------------------------------------//

#pragma region MLP_LOG_CONSOLE

#if !defined(MLP_LOG_CONSOLE)
#define MLP_LOG_CONSOLE
#elif MLP_LOG_CONSOLE == 0
#undef MLP_LOG_CONSOLE
#endif	// MLP_LOG_CONSOLE

constexpr bool console =
#ifdef MLP_LOG_CONSOLE
	true
#else	// !MLP_LOG_CONSOLE
	false
#endif	// MLP_LOG_CONSOLE
	;
#pragma endregion

//-----------------------------------------------------------------//
//--------------------------MLP_LOG_FILE---------------------------//
//-----------------------------------------------------------------//

#pragma region MLP_LOG_FILE

#if !defined(MLP_LOG_FILE)
#define MLP_LOG_FILE
#elif MLP_LOG_FILE == 0
#undef MLP_LOG_FILE
#endif	// MLP_LOG_FILE

constexpr bool file =
#ifdef MLP_LOG_FILE
	true
#else	// !MLP_LOG_FILE
	false
#endif	// MLP_LOG_FILE
	;
#pragma endregion

//-----------------------------------------------------------------//
//------------------------MLP_LOG_LOCATION-------------------------//
//-----------------------------------------------------------------//

#pragma region MLP_LOG_LOCATION

#if !defined(MLP_LOG_LOCATION)
#define MLP_LOG_LOCATION
#elif MLP_LOG_LOCATION == 0
#undef MLP_LOG_LOCATION
#endif	// MLP_LOG_LOCATION

constexpr bool location =
#ifdef MLP_LOG_LOCATION
	true
#else	// !MLP_LOG_LOCATION
	false
#endif	// MLP_LOG_LOCATION
	;
#pragma endregion

}  // namespace log

}  // namespace mlp::def
