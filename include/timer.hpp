#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>

// see 
// https://stackoverflow.com/questions/2808398/easily-measure-elapsed-time

template <
    class result_t = std::chrono::milliseconds,
    class clock_t = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}
#endif
